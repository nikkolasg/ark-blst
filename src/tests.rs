use ark_ff::Field;

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
