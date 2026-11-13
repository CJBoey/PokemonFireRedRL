// fireredsim worker: observation bridge (see obs_bridge.h).
#include "obs_bridge.h"

#include <cstring>

#include "fireredsim_port.h"

#ifndef PN_WORKER_OBS_STUB
#include "observation.h"
#include "skill_events.h"
#include "learned_moves.h"
#endif

namespace pn {

#ifndef PN_WORKER_OBS_STUB

size_t obs_size()
{
    return pn_obs_size();
}

void obs_fill(void *out, int mode)
{
    pn_observe(static_cast<PnObs *>(out), mode);
}

bool obs_waiting()
{
    return pn_obs_waiting() != 0;
}

bool obs_is_stub()
{
    return false;
}

int obs_skill_event_count()
{
    return PN_SKILL_COUNT;
}

void obs_fill_interaction(void *rec, int mode, const PortInteraction &ia)
{
    PnObs *o = static_cast<PnObs *>(rec);
    bool priv = mode == PN_OBS_PRIVILEGED && ia.count != 0;
    o->interact_kind_priv = priv ? ia.kind : 0;
    o->interact_count_priv = priv ? ia.count : 0;
    o->interact_map_group_priv = priv ? ia.mapGroup : 0;
    o->interact_map_num_priv = priv ? ia.mapNum : 0;
    o->interact_local_id_priv = priv ? ia.localId : 0;
    o->interact_x_priv = priv ? ia.x : 0;
    o->interact_y_priv = priv ? ia.y : 0;
}

int obs_interact_fields()
{
    return 1;
}

struct SkillEvents::Slot {
    PnSkillState st;
    bool advanced;     // the env ran a step since the last processed record
    unsigned used;     // PN_ITEM_USED_* since the last processed record: consumables used on the field / in battle
    uint32_t bits;     // the last processed record's events
};

SkillEvents::SkillEvents(int num_envs) : n_(num_envs)
{
    slots_ = new Slot[num_envs > 0 ? num_envs : 1];
    for (int i = 0; i < n_; i++) {
        std::memset(&slots_[i], 0, sizeof(Slot)); // st.valid = 0: the first record initialises
    }
}

SkillEvents::~SkillEvents()
{
    delete[] slots_;
}

void SkillEvents::reset(int i)
{
    pn_skill_reset(&slots_[i].st);
    slots_[i].advanced = false;
    slots_[i].used = 0;
    slots_[i].bits = 0;
}

void SkillEvents::advanced(int i)
{
    slots_[i].advanced = true;
}

void SkillEvents::used_item(int i, int field, int battle)
{
    // (accumulated until a record is processed: a STEP without PN_F_OBS keeps them)
    if (field > 0)
        slots_[i].used |= PN_ITEM_USED_FIELD;
    if (battle > 0)
        slots_[i].used |= PN_ITEM_USED_BATTLE;
}

void SkillEvents::fill(int i, void *rec, int mode)
{
    if (mode != PN_OBS_PRIVILEGED)
        return; // privileged field: 0 in visible mode (pn_observe wrote 0)
    Slot &s = slots_[i];
    PnObs *o = static_cast<PnObs *>(rec);
    if (!s.st.valid || s.advanced) {
        s.bits = pn_skill_record_ex(&s.st, o, s.used);
        s.advanced = false;
        s.used = 0;
    }
    o->skill_events_priv = s.bits;
}

struct LearnedMoves::Slot {
    PnLearnState st;
};

LearnedMoves::LearnedMoves(int num_envs) : n_(num_envs)
{
    slots_ = new Slot[num_envs > 0 ? num_envs : 1];
    for (int i = 0; i < n_; i++)
        std::memset(&slots_[i], 0, sizeof(Slot)); // st.valid = 0: the first record initialises
}

LearnedMoves::~LearnedMoves()
{
    delete[] slots_;
}

void LearnedMoves::reset(int i)
{
    pn_learn_reset(&slots_[i].st);
}

void LearnedMoves::fill(int i, void *rec, int mode, bool step)
{
    if (mode != PN_OBS_PRIVILEGED)
        return; // privileged field: 0 in visible mode (pn_observe wrote 0)
    PnObs *o = static_cast<PnObs *>(rec);
    if (step)
        pn_learn_record_ex(&slots_[i].st, o, o->learned_move_priv, &o->learned_replaced_priv); // (v8)
    else
        pn_learn_init(&slots_[i].st, o); // the field stays 0 (pn_observe wrote 0)
}

int obs_learned_moves_fields()
{
    return 2; // v8: + learned_replaced_priv, summary_cursor
}

int obs_pc_cursor_fields()
{
    return 1; // v9: box_cursor_area / pos / level / is_egg / species / moves
}

int obs_party_cursor_fields()
{
    return 1; // v10: party_cursor, battlers[].party_slot
}

#else // placeholder until obs/ lands

size_t obs_size()
{
    return sizeof(PnObsStub);
}

void obs_fill(void *out, int mode)
{
    PortProbe p;
    Port_Probe(&p);
    PnObsStub o;
    std::memset(&o, 0, sizeof o);
    o.magic = 0x534F4E50u; // "PNOS"
    o.version = 1;
    o.size = sizeof o;
    o.mode = static_cast<uint8_t>(mode ? 1 : 0);
    o.in_battle = p.inBattle;
    o.map_group = static_cast<uint8_t>(p.mapGroup);
    o.map_num = static_cast<uint8_t>(p.mapNum);
    o.x = p.x;
    o.y = p.y;
    o.facing = p.facing;
    o.text_printer_active = p.textPrinterActive;
    o.script_enabled = p.scriptEnabled;
    o.field_controls_locked = p.fieldControlsLocked;
    o.held_keys = p.heldKeys;
    o.new_keys = p.newKeys;
    o.vblank_counter2 = p.vblankCounter2;
    if (mode) {
        o.callback2_priv = p.callback2;
        o.rng_priv = p.rngValue;
    }
    std::memcpy(out, &o, sizeof o);
}

bool obs_waiting()
{
    return true;
}

bool obs_is_stub()
{
    return true;
}

int obs_skill_event_count()
{
    return 0;
}

void obs_fill_interaction(void *, int, const PortInteraction &) {}

int obs_interact_fields()
{
    return 0;
}

struct SkillEvents::Slot {};
SkillEvents::SkillEvents(int num_envs) : n_(num_envs) {}
SkillEvents::~SkillEvents() {}
void SkillEvents::used_item(int, int, int) {}
void SkillEvents::reset(int) {}
void SkillEvents::advanced(int) {}
void SkillEvents::fill(int, void *, int) {}

struct LearnedMoves::Slot {};
LearnedMoves::LearnedMoves(int num_envs) : n_(num_envs) {}
LearnedMoves::~LearnedMoves() {}
void LearnedMoves::reset(int) {}
void LearnedMoves::fill(int, void *, int, bool) {}

int obs_learned_moves_fields()
{
    return 0;
}

int obs_pc_cursor_fields()
{
    return 0;
}

int obs_party_cursor_fields()
{
    return 0;
}

#endif

} // namespace pn
