#ifndef __LANTERN_H__
#define __LANTERN_H__

#ifdef LANTERN_BUILD
#define LANTERN_PTR
#else
#define LANTERN_PTR *
#endif

namespace lantern {

void (LANTERN_PTR print)();

}

#endif