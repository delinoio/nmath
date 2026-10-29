#![allow(dead_code)]

use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

/// A fixture project in its own temporary directory. Dropping it removes only
/// that directory.
pub struct Fixture {
    root: PathBuf,
}

static NEXT: AtomicUsize = AtomicUsize::new(0);

const RUNTIME_DECLARATIONS: &str = "\
export declare function add(a: number, b: number): number;
export declare function sub(a: number, b: number): number;
export declare function mul(a: number, b: number): number;
export declare function neg(a: number): number;
export declare function get<T>(a: readonly T[], i: number): T;
export declare function set<T>(a: T[], i: number, v: T): Promise<void>;
";

impl Fixture {
    /// A project with `src/` sources, `outDir: dist` and a stub
    /// `@delinoio/nmath/nts-runtime` package.
    pub fn new(name: &str) -> Fixture {
        let unique = NEXT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "nmath-compiler-{name}-{}-{unique}",
            std::process::id()
        ));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        let fixture = Fixture {
            root: std::fs::canonicalize(std::env::temp_dir())
                .unwrap()
                .join(root.file_name().unwrap()),
        };
        fixture.write(
            "tsconfig.json",
            r#"{
  // JSONC comments are allowed.
  "compilerOptions": {
    "strict": true,
    "module": "esnext",
    "target": "es2022",
    "moduleResolution": "bundler",
    "outDir": "dist",
    "rootDir": "src",
    "types": []
  },
  "include": ["src"]
}
"#,
        );
        fixture.write(
            "node_modules/@delinoio/nmath/package.json",
            r#"{ "name": "@delinoio/nmath", "version": "0.0.0", "type": "module",
  "exports": { "./nts-runtime": { "types": "./nts-runtime.d.ts", "default": "./nts-runtime.js" } } }"#,
        );
        fixture.write(
            "node_modules/@delinoio/nmath/nts-runtime.d.ts",
            RUNTIME_DECLARATIONS,
        );
        fixture
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn path(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }

    pub fn write(&self, relative: &str, text: &str) {
        let path = self.path(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    pub fn read(&self, relative: &str) -> String {
        std::fs::read_to_string(self.path(relative)).unwrap()
    }

    pub fn tsconfig(&self) -> PathBuf {
        self.path("tsconfig.json")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
