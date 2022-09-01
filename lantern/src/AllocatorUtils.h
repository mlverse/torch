#pragma once

extern const std::thread::id MAIN_THREAD_ID;
extern void (*call_r_gc)(bool);
void wait_for_gc();