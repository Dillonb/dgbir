use std::{cell::RefCell, collections::HashMap, rc::Rc};

use register_type::RegPoolRegister;

use crate::register_allocator::Register;

pub mod register_type {
    use crate::register_allocator::Register;

    pub trait RegPoolRegister {
        fn idx(&self) -> u32;
        fn is_same(r: Register) -> bool;
        fn new(idx: u32) -> Self;
    }

    // This needs to be a separate type than the main Register type because enum variants are not first
    // class types in Rust, so they can't be used as type parameters.
    // https://github.com/rust-lang/rfcs/pull/1450
    // https://github.com/rust-lang/rfcs/pull/2593
    pub struct GPR(u32);
    pub struct SIMD(u32);

    impl RegPoolRegister for GPR {
        fn idx(&self) -> u32 {
            self.0
        }

        fn is_same(r: Register) -> bool {
            match r {
                Register::GPR(_) => true,
                _ => false,
            }
        }

        fn new(idx: u32) -> Self {
            GPR(idx)
        }
    }

    impl RegPoolRegister for SIMD {
        fn idx(&self) -> u32 {
            self.0
        }

        fn is_same(r: Register) -> bool {
            match r {
                Register::SIMD(_) => true,
                _ => false,
            }
        }

        fn new(idx: u32) -> Self {
            SIMD(idx)
        }
    }
}

/// A pool of registers for allocating scratch registers on demand
#[derive(Debug)]
pub struct RegPool {
    pool: Rc<RefCell<RegPoolInternal>>,
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
            pool: Rc::new(RefCell::new(RegPoolInternal::new(regs))),
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

        BorrowedReg {
            reg,
            pool_reg: T::new(reg.index() as u32),
            pool: self.pool.clone(), // Increment refcount
        }
    }

    pub fn reserve<T: RegPoolRegister>(&self, reg: Register) -> BorrowedReg<T> {
        let mut pool = self.pool.borrow_mut();

        if *pool.regs.get(&reg).unwrap_or(&false) {
            panic!("Register already allocated!");
        }

        // If a register is not in the pool, whatever, this pool won't allocate it anyway.
        if pool.regs.contains_key(&reg) {
            pool.regs.insert(reg, true);
        }

        BorrowedReg {
            reg,
            pool_reg: T::new(reg.index() as u32),
            pool: self.pool.clone(), // Increment refcount
        }
    }

    pub fn active_regs(&self) -> Vec<Register> {
        let pool = self.pool.borrow();
        pool.regs
            .iter()
            .filter(|(_, allocated)| **allocated)
            .map(|(reg, _)| *reg)
            .collect()
    }
}

pub struct BorrowedReg<T: RegPoolRegister> {
    reg: Register,
    pub pool_reg: T,
    pool: Rc<RefCell<RegPoolInternal>>,
}

impl<T: RegPoolRegister> BorrowedReg<T> {
    pub fn r(&self) -> u32 {
        self.pool_reg.idx()
    }
}

impl<T: RegPoolRegister> Drop for BorrowedReg<T> {
    fn drop(&mut self) {
        let mut pool = self.pool.borrow_mut();
        if pool.regs.contains_key(&self.reg) {
            pool.regs.insert(self.reg, false);
        }
    }
}
