use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use register_type::RegPoolRegister;

use crate::register_allocator::{Register, RegisterIndex};

pub mod register_type {
    use crate::register_allocator::{Register, RegisterIndex};

    pub trait RegPoolRegister {
        fn idx(&self) -> RegisterIndex;
        fn is_same(r: Register) -> bool;
        fn new(idx: RegisterIndex) -> Self;
    }

    // This needs to be a separate type than the main Register type because enum variants are not first
    // class types in Rust, so they can't be used as type parameters.
    // https://github.com/rust-lang/rfcs/pull/1450
    // https://github.com/rust-lang/rfcs/pull/2593
    pub struct GPR(RegisterIndex);
    pub struct SIMD(RegisterIndex);

    impl RegPoolRegister for GPR {
        fn idx(&self) -> RegisterIndex {
            self.0
        }

        fn is_same(r: Register) -> bool {
            match r {
                Register::GPR(_) => true,
                _ => false,
            }
        }

        fn new(idx: RegisterIndex) -> Self {
            GPR(idx)
        }
    }

    impl RegPoolRegister for SIMD {
        fn idx(&self) -> RegisterIndex {
            self.0
        }

        fn is_same(r: Register) -> bool {
            match r {
                Register::SIMD(_) => true,
                _ => false,
            }
        }

        fn new(idx: RegisterIndex) -> Self {
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
    regs: BTreeMap<Register, bool>,
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
            pool_reg: T::new(reg.index()),
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
            pool_reg: T::new(reg.index()),
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
    pub fn r(&self) -> RegisterIndex {
        self.pool_reg.idx()
    }

    pub fn reg(&self) -> Register {
        self.reg
    }
}

impl<T: RegPoolRegister> Drop for BorrowedReg<T> {
    fn drop(&mut self) {
        let mut pool = self.pool.borrow_mut();
        pool.regs.insert(self.reg, false);
    }
}
