// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! sgl-router: slim KV-aware OpenAI-compatible router for SGLang workers.
//!
//! See `README.md` for build, run, and deployment notes.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod config;
pub mod discovery;
pub mod health;
pub mod policies;
pub mod proxy;
pub mod server;
pub mod tokenizer;
pub mod workers;
