use ark_ec::CurveGroup;
use ark_ec::Group;
use ark_ff::UniformRand;
use criterion::{criterion_group, criterion_main, Bencher, Criterion};

pub fn bench_group_addition<G: Group>(b: &mut Bencher) {
    let r = G::rand(&mut rand::thread_rng());
    let s = G::rand(&mut rand::thread_rng());
    b.iter(|| r + s);
}

pub fn bench_group_multiplication<G: Group>(b: &mut Bencher) {
    let r = G::rand(&mut rand::thread_rng());
    let s = G::ScalarField::rand(&mut rand::thread_rng());
    b.iter(|| r.mul(s));
}

pub fn bench_group_msm<G: CurveGroup>(b: &mut Bencher) {
    let scalars = (0..500)
        .map(|_| G::ScalarField::rand(&mut rand::thread_rng()))
        .collect::<Vec<_>>();
    let bases = (0..500)
        .map(|_| G::Affine::rand(&mut rand::thread_rng()))
        .collect::<Vec<_>>();
    b.iter(|| G::msm(&bases, &scalars).unwrap());
}
fn group_addition_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Group Addition Projective");
    group.bench_function("blst", |b| {
        bench_group_addition::<ark_bls12_381::G1Projective>(b)
    });
    group.bench_function("arkworks", |b| {
        bench_group_addition::<ark_blst::G1Projective>(b)
    });
    group.finish();
}

fn group_multiplication_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Group Multiplication");
    group.bench_function("blst", |b| {
        bench_group_multiplication::<ark_bls12_381::G1Projective>(b)
    });
    group.bench_function("arkworks", |b| {
        bench_group_multiplication::<ark_blst::G1Projective>(b)
    });
    group.finish();
}

fn group_msm_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Group MSM");
    group.bench_function("blst", |b| {
        bench_group_msm::<ark_bls12_381::G1Projective>(b)
    });
    group.bench_function("arkworks", |b| bench_group_msm::<ark_blst::G1Projective>(b));
    group.finish();
}

criterion_group!(
    group,
    group_addition_benchmark,
    group_multiplication_benchmark,
    group_msm_benchmark
);

criterion_main!(group);
