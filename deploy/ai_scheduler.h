/**
 * ai_scheduler.h — C API for the SLM-OS AI scheduling models.
 *
 * This header defines the interface between the SLM-OS kernel/runtime
 * and the trained scheduling models (MLP, XGBoost, RL policy).
 * See plan Section 9.1.
 */

#ifndef AI_SCHEDULER_H
#define AI_SCHEDULER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Number of features in the state vector. */
#define AI_SCHED_STATE_SIZE 108

/** Scheduling action returned by the AI model. */
struct ai_sched_action {
    uint8_t core_assignment;   /* Target core ID, or N_CORES for GPU */
    uint8_t priority_adj;      /* 0=lower, 1=keep, 2=raise */
    uint8_t preempt;           /* 1=preempt current task on target core */
};

/**
 * Initialize the AI scheduler model.
 * Call once during kernel init.
 *
 * @return 0 on success, negative error code on failure.
 */
int ai_scheduler_init(void);

/**
 * Run AI model inference to make a scheduling decision.
 *
 * @param state  Normalized state vector (AI_SCHED_STATE_SIZE floats, [0,1]).
 * @param action Output scheduling action.
 * @return 0 on success, negative error code on failure (caller should
 *         fall back to heuristic scheduler).
 */
int ai_schedule(const float state[AI_SCHED_STATE_SIZE],
                struct ai_sched_action *action);

/**
 * Clean up AI scheduler resources.
 */
void ai_scheduler_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* AI_SCHEDULER_H */
