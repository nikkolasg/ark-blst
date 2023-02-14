use ark_ff::PrimeField;
use criterion::{criterion_group, criterion_main, Bencher, Criterion};

pub fn bench_field_addition<F: PrimeField>(b: &mut Bencher) {
    let r = F::rand(&mut rand::thread_rng());
    let s = F::rand(&mut rand::thread_rng());
    b.iter(|| r + s);
}

pub fn bench_field_multiplication<F: PrimeField>(b: &mut Bencher) {
    let r = F::rand(&mut rand::thread_rng());
    let s = F::rand(&mut rand::thread_rng());
    b.iter(|| r * s);
}

pub fn bench_field_division<F: PrimeField>(b: &mut Bencher) {
    let r = F::rand(&mut rand::thread_rng());
    let s = F::rand(&mut rand::thread_rng());
    b.iter(|| r / s);
}

fn field_addition_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scalar Field Addition");
    group.bench_function("blst", |b| bench_field_addition::<ark_bls12_381::Fr>(b));
    group.bench_function("arkworks", |b| bench_field_addition::<ark_blst::Scalar>(b));
    group.finish();
}

fn field_multiplication_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scalar Field Multiplication");
    group.bench_function("blst", |b| {
        bench_field_multiplication::<ark_bls12_381::Fr>(b)
    });
    group.bench_function("arkworks", |b| {
        bench_field_multiplication::<ark_blst::Scalar>(b)
    });
    group.finish();
}

fn field_division_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scalar Field Division");
    group.bench_function("blst", |b| bench_field_division::<ark_bls12_381::Fr>(b));
    group.bench_function("arkworks", |b| bench_field_division::<ark_blst::Scalar>(b));
    group.finish();
}

criterion_group!(
    benches,
    field_addition_benchmark,
    field_multiplication_benchmark,
    field_division_benchmark,
);
criterion_main!(benches);
