#![cfg(any(feature = "cuda", feature = "opencl"))]

use std::{any::TypeId, ops::AddAssign};

use ark_ec::{AffineRepr, Group};
use ark_ff::Zero;
use ec_gpu::{GpuField, GpuName};
use ec_gpu_gen::{
    rust_gpu_tools::{program_closures, Device, Program},
    EcError,
};

use crate::{fp::Fp, fp12::Fp12, fp2::Fp2, fp6::Fp6, g1::G1Affine, g2::G2Affine, scalar::Scalar};

/// On the GPU, the exponents are split into windows, this is the maximum number of such windows.
const MAX_WINDOW_SIZE: usize = 10;
/// In CUDA this is the number of blocks per grid (grid size).
const LOCAL_WORK_SIZE: usize = 128;
/// Let 20% of GPU memory be free, this is an arbitrary value.
const MEMORY_PADDING: f64 = 0.2f64;
/// The Nvidia Ampere architecture is compute capability major version 8.
const AMPERE: u32 = 8;

/// Divide and ceil to the next value.
const fn div_ceil(a: usize, b: usize) -> usize {
    if a % b == 0 {
        a / b
    } else {
        (a / b) + 1
    }
}

/// The number of units the work is split into. One unit will result in one CUDA thread.
///
/// Based on empirical results, it turns out that on Nvidia devices with the Ampere architecture,
/// it's faster to use two times the number of work units.
const fn work_units(compute_units: u32, compute_capabilities: Option<(u32, u32)>) -> usize {
    match compute_capabilities {
        Some((AMPERE, _)) => LOCAL_WORK_SIZE * compute_units as usize * 2,
        _ => LOCAL_WORK_SIZE * compute_units as usize,
    }
}

/// Multiexp kernel for a single GPU.
struct SingleMultiexpKernel<'a, G>
where
    G: AffineRepr,
{
    program: Program,
    /// The number of exponentiations the GPU can handle in a single execution of the kernel.
    _max_exponentiations: usize,
    /// The number of units the work is split into. It will results in this amount of threads on
    /// the GPU.
    work_units: usize,
    /// An optional function which will be called at places where it is possible to abort the
    /// multiexp calculations. If it returns true, the calculation will be aborted with an
    /// [`EcError::Aborted`].
    maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,

    _phantom: std::marker::PhantomData<G>,
}

/// Calculates the maximum number of terms that can be put onto the GPU memory.
fn calc_chunk_size<G>(mem: u64, work_units: usize) -> usize
where
    G: AffineRepr,
    G::ScalarField: ark_ff::PrimeField,
{
    let aff_size = std::mem::size_of::<G>();
    let exp_size = exp_size::<G::ScalarField>();
    let proj_size = std::mem::size_of::<G::Group>();

    // Leave `MEMORY_PADDING` percent of the memory free.
    let max_memory = ((mem as f64) * (1f64 - MEMORY_PADDING)) as usize;
    // The amount of memory (in bytes) of a single term.
    let term_size = aff_size + exp_size;
    // The number of buckets needed for one work unit
    let max_buckets_per_work_unit = 1 << MAX_WINDOW_SIZE;
    // The amount of memory (in bytes) we need for the intermediate steps (buckets).
    let buckets_size = work_units * max_buckets_per_work_unit * proj_size;
    // The amount of memory (in bytes) we need for the results.
    let results_size = work_units * proj_size;

    (max_memory - buckets_size - results_size) / term_size
}

/// The size of the exponent in bytes.
///
/// It's the actual bytes size it needs in memory, not it's theoratical bit size.
fn exp_size<F: ark_ff::PrimeField>() -> usize {
    std::mem::size_of::<F::BigInt>()
}
impl<'a, G> SingleMultiexpKernel<'a, G>
where
    G: AffineRepr + GpuName,
{
    /// Create a new Multiexp kernel instance for a device.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    fn create(
        program: Program,
        device: &Device,
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> Result<Self, EcError> {
        let mem = device.memory();
        let compute_units = device.compute_units();
        let compute_capability = device.compute_capability();
        let work_units = work_units(compute_units, compute_capability);
        let chunk_size = calc_chunk_size::<G>(mem, work_units);

        Ok(SingleMultiexpKernel {
            program,
            _max_exponentiations: chunk_size,
            work_units,
            maybe_abort,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Run the actual multiexp computation on the GPU.
    ///
    /// The number of `bases` and `exponents` are determined by [`SingleMultiexpKernel`]`::n`, this
    /// means that it is guaranteed that this amount of calculations fit on the GPU this kernel is
    /// running on.
    fn multiexp(
        &self,
        bases: &[G],
        exponents: &[<G::ScalarField as ark_ff::PrimeField>::BigInt],
    ) -> Result<G::Group, EcError> {
        assert_eq!(bases.len(), exponents.len());

        if let Some(maybe_abort) = &self.maybe_abort {
            if maybe_abort() {
                return Err(EcError::Aborted);
            }
        }
        let window_size = self.calc_window_size(bases.len());
        // windows_size * num_windows needs to be >= 256 in order for the kernel to work correctly.
        let num_windows = div_ceil(256, window_size);
        let num_groups = self.work_units / num_windows;
        let bucket_len = 1 << window_size;

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let closures = program_closures!(|program, _arg| -> Result<Vec<G::Group>, EcError> {
            let base_buffer = program.create_buffer_from_slice(&bases)?;
            let exp_buffer = program.create_buffer_from_slice(exponents)?;

            // It is safe as the GPU will initialize that buffer
            let bucket_buffer =
                unsafe { program.create_buffer::<G::Group>(self.work_units * bucket_len)? };
            // It is safe as the GPU will initialize that buffer
            let result_buffer = unsafe { program.create_buffer::<G::Group>(self.work_units)? };

            // The global work size follows CUDA's definition and is the number of
            // `LOCAL_WORK_SIZE` sized thread groups.
            let global_work_size = div_ceil(num_windows * num_groups, LOCAL_WORK_SIZE);

            // NOTE vmx 2023-02-10: This is a hack. We currently use a kernel created with the
            // blstrs types. Hence we need to call those specific functions.
            //let kernel_name = format!("{}_multiexp", G::name());
            let kernel_name = if TypeId::of::<G>() == TypeId::of::<G1Affine>() {
                "blstrs__g1__G1Affine_multiexp"
            } else if TypeId::of::<G>() == TypeId::of::<G2Affine>() {
                "blstrs__g2__G2Affine_multiexp"
            } else {
                panic!("unknown type")
            };
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&base_buffer)
                .arg(&bucket_buffer)
                .arg(&result_buffer)
                .arg(&exp_buffer)
                .arg(&(bases.len() as u32))
                .arg(&(num_groups as u32))
                .arg(&(num_windows as u32))
                .arg(&(window_size as u32))
                .run()?;

            let mut results = vec![G::Group::zero(); self.work_units];
            program.read_into_buffer(&result_buffer, &mut results)?;

            Ok(results)
        });

        let results = self.program.run(closures, ())?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = G::Group::zero();
        let mut bits = 0;
        let exp_bits = exp_size::<G::ScalarField>() * 8;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc = acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }

    /// Calculates the window size, based on the given number of terms.
    ///
    /// For best performance, the window size is reduced, so that maximum parallelism is possible.
    /// If you e.g. have put only a subset of the terms into the GPU memory, then a smaller window
    /// size leads to more windows, hence more units to work on, as we split the work into
    /// `num_windows * num_groups`.
    fn calc_window_size(&self, num_terms: usize) -> usize {
        // The window size was determined by running the `gpu_multiexp_consistency` test and
        // looking at the resulting numbers.
        let window_size = ((div_ceil(num_terms, self.work_units) as f64).log2() as usize) + 2;
        std::cmp::min(window_size, MAX_WINDOW_SIZE)
    }
}

pub(crate) fn msm<G>(
    bases: &[G],
    exponents: &[<G::ScalarField as ark_ff::PrimeField>::BigInt],
) -> Result<G::Group, EcError>
where
    G: AffineRepr + GpuName,
{
    let devices = Device::all();
    let device = devices[0];
    let program = ec_gpu_gen::program!(device).expect("Cannot create program!");
    let kernel = SingleMultiexpKernel::<G>::create(program, &device, None)
        .expect("Cannot initialize kernel!");
    // TODO vmx 2023-02-08: Add logic to split the bases an exponents into chunks, in case the GPU
    // doesn't have enough memory.
    kernel.multiexp(bases, exponents)
}

macro_rules! impl_gpu_name {
    ($type:ty) => {
        impl GpuName for $type {
            fn name() -> String {
                ec_gpu::name!()
            }
        }
    };
}

macro_rules! impl_gpu_field {
    ($type:ident) => {
        impl GpuField for $type {
            fn one() -> Vec<u32> {
                blstrs::$type::one()
            }

            fn r2() -> Vec<u32> {
                blstrs::$type::r2()
            }

            fn modulus() -> Vec<u32> {
                blstrs::$type::modulus()
            }

            fn sub_field_name() -> Option<String> {
                blstrs::$type::sub_field_name()
            }
        }
    };
}

impl_gpu_name!(Fp);
impl_gpu_name!(Fp2);
impl_gpu_name!(Fp6);
impl_gpu_name!(Fp12);
impl_gpu_name!(G1Affine);
impl_gpu_name!(G2Affine);
impl_gpu_name!(Scalar);

impl_gpu_field!(Fp);
impl_gpu_field!(Fp2);
impl_gpu_field!(Scalar);
