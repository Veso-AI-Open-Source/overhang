//! Integration test against a live overhangd (only runs when one is up on 11544).
//! Run (from daemon/): cargo run -- --config tests/overhangd.mock.toml &
//! then (from app/):   cargo test -- --ignored

use overhang_app::api::{self, Cmd};
use std::time::{Duration, Instant};

#[test]
#[ignore = "needs a daemon (real or mock) on 127.0.0.1:11544"]
fn status_and_chat_stream() {
    let client = api::start(|| {});
    client.send(Cmd::RefreshStatus);
    client.send(Cmd::SendChat("why does a 235B model fit in 24GB?".into()));

    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        std::thread::sleep(Duration::from_millis(200));
        let s = client.shared.lock().unwrap();
        // the repo mock engine streams exactly 3 tokens ("hello , world")
        let streamed = s
            .messages
            .iter()
            .any(|m| m.role == "assistant" && !m.content.is_empty());
        if s.daemon_up && s.capacity.is_some() && streamed && !s.generating {
            let cap = s.capacity.as_ref().unwrap();
            assert!(!cap.models.is_empty(), "capacity ladder should have rows");
            println!("assistant: {}", s.messages.last().unwrap().content);
            println!("tok/s at end: {:.1}", s.gen_tok_s);
            return;
        }
        assert!(Instant::now() < deadline, "timed out; state: daemon_up={} cap={} msgs={:?}",
            s.daemon_up, s.capacity.is_some(), s.messages);
    }
}

#[test]
#[ignore = "needs a daemon (real or mock) on 127.0.0.1:11544"]
fn system_discovery_and_load() {
    let client = api::start(|| {});
    client.send(Cmd::RefreshStatus);

    // discovery: /system must report a measured machine, nothing hardcoded
    let deadline = Instant::now() + Duration::from_secs(30);
    let first_model = loop {
        std::thread::sleep(Duration::from_millis(200));
        let s = client.shared.lock().unwrap();
        if let (Some(sys), Some(cap)) = (&s.system, &s.capacity) {
            assert!(!sys.chip.is_empty(), "chip should be discovered");
            assert!(sys.total_ram_gb > 0.0, "RAM should be discovered");
            assert!(sys.logical_cores > 0, "cores should be discovered");
            assert!(!cap.models.is_empty(), "ladder should list installed containers");
            break cap.models[0].name.clone();
        }
        assert!(Instant::now() < deadline, "no /system report; unsupported={}",
            s.system_unsupported);
    };

    // load workflow: engine spins up on the container, /status reconciles by name
    client.send(Cmd::Load(first_model.clone()));
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        std::thread::sleep(Duration::from_millis(200));
        let s = client.shared.lock().unwrap();
        assert!(!s.load_unsupported, "daemon should support /engine/load");
        let active = s.capacity.as_ref().is_some_and(|c| {
            c.engine_up && c.models.iter().any(|m| m.active && m.name == first_model)
        });
        if s.loading_model.is_none() && active {
            return;
        }
        assert!(Instant::now() < deadline, "load did not reconcile; cap={:?}", s.capacity);
    }
}
