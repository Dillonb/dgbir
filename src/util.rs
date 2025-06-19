use dynasmrt::cache_control::synchronize_icache;

pub fn flush_icache(slice: &[u8]) {
    synchronize_icache(slice);
}
