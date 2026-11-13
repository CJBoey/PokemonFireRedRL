// fireredsim worker: the structured observation of the active environment.
//
// With the observation library (obs/observation.h, target pnobs) this forwards
// to pn_obs_size()/pn_observe(). Without it (PN_WORKER_OBS_STUB) a small
// placeholder record built from Port_Probe is used, so that the worker, the
// Python package and the tests work before obs/ exists. The placeholder is NOT
// the observation contract of docs/OBSERVATION.md.
#pragma once

#include <cstddef>
#include <cstdint>

struct PortInteraction; // fireredsim_port.h

namespace pn {

size_t obs_size();
// Fills obs_size() bytes from the active environment; mode 0 visible, 1 privileged.
void obs_fill(void *out, int mode);
// waiting_for_input of the active environment (pn_obs_waiting: the observer's predicate without
// building the record); the placeholder has no predicate and always returns true.
bool obs_waiting();
// True while the placeholder is compiled in.
bool obs_is_stub();
// The number of skill event bits (0 .. n-1 of PnObs.skill_events_priv) this worker sets: PN_SKILL_COUNT (13 since
// reward v0.12's mart event, 14 since the use_consumable event of 2026-10-28, 15 since use_battle_item and 16 since
// mart_buy, both 2026-10-29; the placeholder: 0). Reported as
// "skill_events" by --info: the bits were added without a record layout or version change, so Python learns from
// it which bits are ever set (an observer v5 worker without the key sets bits 0-11 only).
int obs_skill_event_count();

// Observer v6 PnObs.interact_*_priv (reward v0.14): the overworld interactions the game reported
// (fireredsim_port.h Port_TakeInteraction) during the frames of the op that produced the record. The
// worker clears the latch before every op and takes it after a STEP op; `ia` is that STEP's take (all
// zero for every other op). Privileged mode: the fields from `ia`; visible mode: 0 (as pn_observe wrote
// them). The placeholder observation has no such fields (no-op).
void obs_fill_interaction(void *rec, int mode, const PortInteraction &ia);
// "interact" of --info: 1 if the record has the v6 interaction fields (0: the placeholder).
int obs_interact_fields();

// Observer v5 PnObs.skill_events_priv (obs/skill_events.h): reward v0.11's skill-retention events,
// detected per env against the env's previous record. One state per env of the worker:
//   reset(i)     a reset / snapshot restore of env i: its next record initialises the state, 0 events
//   advanced(i)  env i ran a step: its next record is a new one
//   used_item(i, field, battle)  the game consumed `field` items by uses outside battle and `battle` by uses in
//                battle during env i's step (fireredsim_port.h Port_TakeItemUse after the STEP's frames): the
//                next processed record of env i carries PN_SKILL_USE_CONSUMABLE / PN_SKILL_USE_BATTLE_ITEM
//                (obs/skill_events.h pn_skill_record_ex)
//   fill(i, rec, mode)  after obs_fill of env i: writes the record's skill_events_priv (privileged mode;
//                in visible mode the field stays 0). A record of a state that did not change since
//                the last one (OBSERVE / SNAPSHOT ops) is not processed again: it repeats that
//                record's bits. The placeholder observation has no such field (no-op).
class SkillEvents {
public:
    explicit SkillEvents(int num_envs);
    ~SkillEvents();
    SkillEvents(const SkillEvents &) = delete;
    SkillEvents &operator=(const SkillEvents &) = delete;
    void reset(int i);
    void advanced(int i);
    void used_item(int i, int field, int battle);
    void fill(int i, void *rec, int mode);

private:
    struct Slot;
    Slot *slots_ = nullptr;
    int n_ = 0;
};

// Observer v7 PnObs.learned_move_priv (obs/learned_moves.h): the move each party slot's mon gained during the
// STEP that produced the record, against the env's last processed record (mons matched by personality and OT id).
// One state per env of the worker:
//   reset(i)     a reset / snapshot restore of env i: its next record initialises the state
//   fill(i, rec, mode, step)  after obs_fill of env i: a STEP's record (step) reports its gains and becomes the
//                state; any other op's record (OBSERVE / SNAPSHOT / RESET / RESTORE) reports 0 and only
//                initialises an uninitialised state. Visible mode: the field stays 0 (pn_observe wrote 0) and the
//                state is not touched. The placeholder observation has no such field (no-op).
class LearnedMoves {
public:
    explicit LearnedMoves(int num_envs);
    ~LearnedMoves();
    LearnedMoves(const LearnedMoves &) = delete;
    LearnedMoves &operator=(const LearnedMoves &) = delete;
    void reset(int i);
    void fill(int i, void *rec, int mode, bool step);

private:
    struct Slot;
    Slot *slots_ = nullptr;
    int n_ = 0;
};
// "learned_moves" of --info: 1 if the record has the v7 fields offered_move / offered_slot / learned_move_priv,
// 2 with observer v8's learned_replaced_priv / summary_cursor (0: the placeholder).
int obs_learned_moves_fields();
// "pc_cursor" of --info: 1 if the record has observer v9's box_cursor_* fields (the mon under the storage
// system's cursor, docs/PC_OBS.md; 0: older layouts / the placeholder).
int obs_pc_cursor_fields();
// "party_cursor" of --info: 1 if the record has observer v10's party_cursor and battlers[].party_slot
int obs_party_cursor_fields();

#pragma pack(push, 1)
// Placeholder layout (python/fireredsim/obs.py STUB_DTYPE mirrors it).
struct PnObsStub {
    uint32_t magic;        // 'PNOS'
    uint16_t version;      // 1
    uint16_t size;         // sizeof(PnObsStub)
    uint8_t mode;          // 0 visible, 1 privileged
    uint8_t in_battle;
    uint8_t map_group, map_num;
    int16_t x, y;
    uint8_t facing, text_printer_active, script_enabled, field_controls_locked;
    uint16_t held_keys, new_keys;
    uint32_t vblank_counter2;
    uint32_t callback2_priv; // privileged
    uint32_t rng_priv;       // privileged
    uint8_t _pad[28];
};
#pragma pack(pop)
static_assert(sizeof(PnObsStub) == 64, "stub observation layout");

} // namespace pn
