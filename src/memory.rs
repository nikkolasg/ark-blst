pub(crate) fn slice_to_constant_size<const N: usize>(v: &[u8]) -> &[u8; N] {
    let ptr = v.as_ptr() as *const [u8; N];
    unsafe { &*ptr }
}

pub(crate) fn constant_size_to_slice<const N: usize>(v: &[u8; N]) -> &[u8] {
    let ptr = v as *const [u8; N] as *const [u8];
    unsafe { &*ptr }
}
