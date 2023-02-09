use ark_ec::{
    models::CurveConfig,
    pairing::{MillerLoopOutput, Pairing, PairingOutput},
};
use ark_ff::{CyclotomicMultSubgroup, One, Zero};
use group::prime::PrimeCurveAffine;

use crate::{
    fp12::Fp12,
    g1::{Config as G1Config, G1Affine, G1Projective},
    g2::{G2Affine, G2Prepared, G2Projective},
};

impl CyclotomicMultSubgroup for Fp12 {
    const INVERSE_IS_FAST: bool = true;

    fn cyclotomic_inverse_in_place(&mut self) -> Option<&mut Self> {
        if self.is_zero() {
            None
        } else {
            blstrs::Fp12::from(*self).conjugate();
            Some(self)
        }
    }

    fn cyclotomic_square_in_place(&mut self) -> &mut Self {
        let mut out = blst::blst_fp12::default();
        unsafe { blst::blst_fp12_cyclotomic_sqr(&mut out, &blstrs::Fp12::from(*self).into()) };
        *self = Fp12::from(blstrs::Fp12::from(out));
        self
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Bls12;

impl Pairing for Bls12 {
    type BaseField = <G1Config as CurveConfig>::BaseField;
    type ScalarField = <G1Config as CurveConfig>::ScalarField;
    type G1 = G1Projective;
    type G1Affine = G1Affine;
    type G1Prepared = G1Affine;
    type G2 = G2Projective;
    type G2Affine = G2Affine;
    type G2Prepared = G2Prepared;
    type TargetField = Fp12;

    // This implementation is a copy from `blstrs`.
    fn multi_miller_loop(
        a: impl IntoIterator<Item = impl Into<Self::G1Prepared>>,
        b: impl IntoIterator<Item = impl Into<Self::G2Prepared>>,
    ) -> MillerLoopOutput<Self> {
        let mut res = blst::blst_fp12::default();

        for (i, (p, q)) in a.into_iter().zip(b).enumerate() {
            let (p, q): (blstrs::G1Affine, blstrs::G2Prepared) = (p.into().into(), q.into().into());
            let mut tmp = blst::blst_fp12::default();
            if (p.is_identity() | q.is_identity()).into() {
                // Define pairing with zero as one, matching what `pairing` does.
                tmp = blstrs::Fp12::from(Fp12::one()).into();
            } else {
                blstrs::miller_loop_lines(&mut tmp, &q, &p);
            }
            if i == 0 {
                res = tmp;
            } else {
                unsafe {
                    blst::blst_fp12_mul(&mut res, &res, &tmp);
                }
            }
        }

        MillerLoopOutput(Fp12::from(blstrs::Fp12::from(res)))
    }

    fn final_exponentiation(f: MillerLoopOutput<Self>) -> Option<PairingOutput<Self>> {
        Some(PairingOutput(Fp12::from(blstrs::final_exponentiation(
            &f.0.into(),
        ))))
    }
}

#[cfg(test)]
mod test {
    use crate::scalar::Scalar;

    use super::*;
    use ark_ec::CurveGroup;
    use ark_ec::Group;
    use ark_ff::UniformRand;
    #[test]
    fn pairing() {
        let s1 = Scalar::rand(&mut rand::thread_rng());
        let p1s = G1Projective::generator() * s1;
        let p2 = G2Projective::generator();
        let p1 = G1Projective::generator();
        let p2s = G2Projective::generator() * s1;
        let left = <Bls12 as Pairing>::pairing(p1s.into_affine(), p2.into_affine());
        let right = <Bls12 as Pairing>::pairing(p1.into_affine(), p2s.into_affine());
        assert!(left == right)
    }
}
