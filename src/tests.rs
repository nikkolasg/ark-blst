use ark_ec::CurveGroup;
use ark_ec::VariableBaseMSM;
use ark_ff::Field;
use ark_ff::UniformRand;
use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
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

pub trait Serializable: Eq + UniformRand + CanonicalSerialize + CanonicalDeserialize {}

impl<T> Serializable for T where T: Eq + UniformRand + CanonicalSerialize + CanonicalDeserialize {}
pub fn compatibility<E1: Serializable, E2: Serializable>() {
    let e1 = E1::rand(&mut rand::thread_rng());
    let e2 = E2::rand(&mut rand::thread_rng());

    // Check if reading serialized element of E1 via E2 gives back same serialization
    let mut v1 = Vec::new();
    e1.serialize_compressed(&mut v1).expect("this should work");
    let read_1 = E2::deserialize_compressed(&v1[..]).expect("deserialize doesn't work");
    let mut written_1 = Vec::new();
    read_1
        .serialize_compressed(&mut written_1)
        .expect("serializable didnt work");
    assert_eq!(v1, written_1);

    // Check if reading serialized element of E2 via E1 gives back same serialization
    let mut v2 = Vec::new();
    e2.serialize_compressed(&mut v2).expect("this should work");
    let read_2 = E1::deserialize_compressed(&v2[..]).expect("deserialize doesn't work");
    let mut written_2 = Vec::new();
    read_2
        .serialize_compressed(&mut written_2)
        .expect("serializable didnt work");
    assert_eq!(v2, written_2);
}
