#pragma once

/* Thesis evaluation parameters*/
constexpr int IMG_H = 96;
constexpr int IMG_W = 320;
constexpr int DISP  = 32;
constexpr int WIN   = 3;
constexpr int MED_WIN = 3;
constexpr int MED_RAD = MED_WIN/2;
static_assert(MED_WIN % 2 == 1, "MED_WIN must be odd");


constexpr int CENSUS_WIN = WIN;
constexpr int CENSUS_CX = CENSUS_WIN >> 1;
constexpr int CENSUS_CY = CENSUS_WIN >> 1;


static constexpr int RIGHT_STRIPE_W = DISP + WIN - 1;
