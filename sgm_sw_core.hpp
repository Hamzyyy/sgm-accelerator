#ifndef SGM_SW_CORE_HPP
#define SGM_SW_CORE_HPP

#include <cstdint>
#include "sgm_config.hpp"

static const uint16_t P1_core = 10;
static const uint16_t P2_core = 150;
static const uint16_t INF_COST_core = 4095;

void sgm_sw_core(
		const uint8_t left[IMG_H][IMG_W],
		const uint8_t right[IMG_H][IMG_W],
		uint8_t disp[IMG_H][IMG_W]);

#endif
