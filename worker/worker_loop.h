// fireredsim worker loop (docs/IPC_PROTOCOL.md, pn_ipc.h).
#pragma once

namespace pn {

// Command line of fireredsim-worker (argv[0] = program name):
//   --info                                     print the JSON description of this executable
//   (--shm-fd FD | --shm-path PATH) --worker W [--template SNAPSHOT]... [--template-script SCRIPT]...
//   [--bedroom-cache N] [--save-cache N] [--cpu C] [--quiet-faults] [--preset-flag ID]... [--preset-var ID=VALUE]...
//     --template FILE          snapshot file: template PN_TEMPLATE_FILE0 + i
//     --template-script FILE   input script replayed from power-on: PN_TEMPLATE_SCRIPT0 + i
//                              (directives "#! frames N", "#! timer1 A,B" in comments)
//     --bedroom-cache N        start states cached per script template (LRU)
//     --save-cache N           save start states cached (all saves together, LRU)
//     --quiet-faults           no per-site retail-fault notices on stderr
//     --preset-flag ID         new-game preset (envpool.h): set event flag ID (1..0x8FF) in every
//                              script template's start (bedroom included), before the start cache
//     --preset-var ID=VALUE    new-game preset: game variable ID (0x4000..0x40FF) = VALUE (0..65535)
//                              (numbers decimal or 0x hex; both repeatable; --info "preset" lists the ranges)
// Serves worker block W until SHUTDOWN or until the parent dies; returns the exit code.
//   --save-start SAV [--seed A,B] [--script-out FILE]    save_start_tool_main
//   --replay [--sav SAV] [--seed A,B] [--input SCRIPT] [--frames N] [--dump-state FILE]
//                                                        replay_tool_main
int worker_main(int argc, char **argv);

// fireredsim-worker --save-start SAV [--seed A,B,...] [--script-out FILE]
//   Runs the save start of SAV (EnvPool, PN_TEMPLATE_SAVE: power-on with the save, CONTINUE
//   driver) with those timer-1 values (default 0) and prints a JSON summary (frames, state
//   hash, map, position); --script-out writes the driver's buttons as an input script
//   (tests/inputs/FORMAT.md, "#! frames N" directive) that fireredsim-cli --sav SAV --seed
//   A,B --input FILE --frames N replays.
int save_start_tool_main(int argc, char **argv);

// fireredsim-worker --replay [--sav SAV] [--seed A,B,...] [--input SCRIPT] [--frames N]
//                            [--dump-state FILE]
//   fireredsim-cli's path (Simulator::create_env with the flash image and timer-1 values,
//   then the script's buttons frame by frame) inside this executable: prints
//   "done: N frames, final hash H" with H comparable to the worker's state hashes
//   (PokeVecEnv.state_hash; hashes of different executables differ: native code addresses
//   are part of the state). --frames 0/default: up to "#! frames N" if the script has that
//   directive, else the end of its last range.
int replay_tool_main(int argc, char **argv);

} // namespace pn
