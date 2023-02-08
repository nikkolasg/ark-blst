use ark_ec::CurveGroup;
use ark_ec::VariableBaseMSM;
use ark_ff::Field;
use ark_ff::UniformRand;
use std::ops::Neg;

pub fn field_test<F: Field>() {
    let r = F::rand(&mut rand::thread_rng());
    let s = F::rand(&mut rand::thread_rng());
    let rps = r + s;
    assert!(rps.neg() + rps == F::zero());
    let spr = s + r;
    assert_eq!(rps, spr);

    let rps = r * s;
    assert!(rps.div(rps) == F::one());
    let spr = s * r;
    assert_eq!(rps, spr);

    let mut buff = Vec::new();
    r.serialize_compressed(&mut buff).unwrap();
    let r2 = F::deserialize_compressed(&buff[..]).unwrap();
    assert_eq!(r, r2);
}
use ark_ff::PrimeField;
pub fn group_test<G: CurveGroup>() {
    let r = G::rand(&mut rand::thread_rng());
    let s = G::rand(&mut rand::thread_rng());
    let rps = r + s;
    assert!(rps.neg() + rps == G::zero());
    let spr = s + r;
    assert_eq!(rps, spr);

    let scalar = G::ScalarField::rand(&mut rand::thread_rng());
    let rs = r.mul(scalar);
    assert!(rs + rs.neg() == G::zero());
    let mrs = r.mul(scalar.neg());
    assert!(rs + mrs == G::zero());

    let r1 = r.mul_bigint(scalar.into_bigint());
    assert!(rs == r1);

    let mut buff = Vec::new();
    r.serialize_compressed(&mut buff).unwrap();
    let r2 = G::deserialize_compressed(&buff[..]).unwrap();
    assert_eq!(r, r2);

    // --- MSM part
    let scalars = (0..10)
        .map(|_| G::ScalarField::rand(&mut rand::thread_rng()))
        .collect::<Vec<_>>();
    let bases = (0..10)
        .map(|_| G::rand(&mut rand::thread_rng()))
        .collect::<Vec<_>>();
    // manual msm
    let exp = scalars
        .iter()
        .zip(bases.iter())
        .fold(G::zero(), |acc, (s, b)| acc + b.mul(*s));
    assert_eq!(scalars.len(), bases.len());
    let affines = G::normalize_batch(&bases);
    assert_eq!(scalars.len(), affines.len());
    // msm from crate
    let res = <G as VariableBaseMSM>::msm(&affines, &scalars).unwrap();
    assert_eq!(exp, res);
}
