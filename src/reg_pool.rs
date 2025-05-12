// This needs to be a separate type than the main Register type because enum variants are not first
// class types in Rust, so they can't be used as type parameters.
// https://github.com/rust-lang/rfcs/pull/1450
// https://github.com/rust-lang/rfcs/pull/2593

use std::{cell::RefCell, collections::HashMap, sync::Arc};

use register_type::RegPoolRegister;

use crate::register_allocator::Register;

pub mod register_type {
    use crate::register_allocator::Register;

    pub trait RegPoolRegister {
        fn idx(&self) -> u32;
        fn is_same(r: Register) -> bool;
        fn new(idx: u32) -> Self;
    }

    pub struct GPR(u32);

    impl RegPoolRegister for GPR {
        fn idx(&self) -> u32 {
            self.0
        }

        fn is_same(r: Register) -> bool {
            match r {
                Register::GPR(_) => true,
            }
        }

        fn new(idx: u32) -> Self {
            GPR(idx)
        }
    }
}

/// A pool of registers for allocating scratch registers on demand
#[derive(Debug)]
pub struct RegPool {
    pool: Arc<RefCell<RegPoolInternal>>,
}

#[derive(Debug)]
struct RegPoolInternal {
    regs: HashMap<Register, bool>,
}

impl RegPoolInternal {
    fn new(regs: Vec<Register>) -> Self {
        RegPoolInternal {
            regs: regs.iter().map(|r| (*r, false)).collect(),
        }
    }
}

impl RegPool {
    pub fn new(regs: Vec<Register>) -> Self {
        RegPool {
            pool: Arc::new(RefCell::new(RegPoolInternal::new(regs))),
        }
    }

    pub fn borrow<T: RegPoolRegister>(&self) -> BorrowedReg<T> {
        let mut pool = self.pool.borrow_mut();

        let maybe_reg = pool
            .regs
            .iter()
            .filter(|(_, allocated)| !(**allocated))
            .map(|(reg, _)| reg)
            .find(|reg| T::is_same(**reg));

        if maybe_reg.is_none() {
            panic!("No registers found!");
        }

        let reg = *maybe_reg.unwrap();
        pool.regs.insert(reg, true);

        println!("Borrowed temp reg {}", reg);

        BorrowedReg {
            reg,
            pool_reg: T::new(reg.index() as u32),
            pool: self.pool.clone(), // Increment refcount
        }
    }
}

pub struct BorrowedReg<T: RegPoolRegister> {
    reg: Register,
    pub pool_reg: T,
    pool: Arc<RefCell<RegPoolInternal>>,
}

impl<T: RegPoolRegister> BorrowedReg<T> {
    pub fn r(&self) -> u32 {
        self.pool_reg.idx()
    }
}

impl<T: RegPoolRegister> Drop for BorrowedReg<T> {
    fn drop(&mut self) {
        println!("Returning temp reg {} to the pool", self.reg);
        self.pool.borrow_mut().regs.insert(self.reg, false);
    }
}
