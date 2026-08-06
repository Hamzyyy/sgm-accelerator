#ifndef SGM_SW_CORE_HPP
#define SGM_SW_CORE_HPP

#include <stdint.h>
#include "sgm_config.hpp"

extern uint64_t t_linebuffer;
extern uint64_t t_slidingwindow;
extern uint64_t t_computecensus;
extern uint64_t t_aggregatecost;
extern uint64_t t_commitcosts;

static const uint16_t P1_core = 3;
static const uint16_t P2_core = 15;
static const uint16_t INF_COST_core = 4095;

void sgm_sw_core(
		const uint8_t left[IMG_H][IMG_W],
		const uint8_t right[IMG_H][IMG_W],
		uint8_t disp[IMG_H][IMG_W]);

#endif
