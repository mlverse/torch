#ifndef __LANTERN_H__
#define __LANTERN_H__

#ifndef _WIN32
#include <dlfcn.h>
#else
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#endif

#ifdef LANTERN_BUILD
#define LANTERN_PTR
#ifdef _WIN32
#define LANTERN_API extern "C" __declspec(dllexport)
#endif
#else
#define LANTERN_PTR *
#endif

#ifndef LANTERN_API
#define LANTERN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif
  
inline const char* pathSeparator()
{
#ifdef _WIN32
    return "\\";
#else
    return "/";
#endif
}
  
inline const char* libraryName()
{
#ifdef __APPLE__
  return "liblantern.dylib";
#else
#ifdef _WIN32
  return "lantern.dll";
#else
  return "liblantern.so";
#endif
#endif
}
  
LANTERN_API void (LANTERN_PTR lanternTest)();
  
#ifdef __cplusplus
}
#endif

#ifndef LANTERN_BUILD

void* pLibrary = NULL;

#define LOAD_SYMBOL(name)                                     \
if (!laternLoadSymbol(pLibrary, #name, (void**) &name, pError))     \
  return false;

void lanternLoadError(std::string* pError)
{
#ifdef _WIN32
  LPVOID lpMsgBuf;
  DWORD dw = ::GetLastError();
  
  DWORD length = ::FormatMessage(
    FORMAT_MESSAGE_ALLOCATE_BUFFER |
      FORMAT_MESSAGE_FROM_SYSTEM |
      FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL,
      dw,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPTSTR) &lpMsgBuf,
      0, NULL );
  
  if (length != 0)
  {
    std::string msg((LPTSTR)lpMsgBuf);
    LocalFree(lpMsgBuf);
    pError->assign(msg);
  }
  else
  {
    pError->assign("Unknown error");
  }
#else
  const char* msg = ::dlerror();
  if (msg != NULL)
    pError->assign(msg);
  else
    pError->assign("Unknown error");
#endif
}

bool lanternLoadLibrary(const std::string& libPath, std::string* pError)
{
  pLibrary = NULL;

  std::string separator = (libPath.back() == '/' || libPath.back() == '\\') ? "" : pathSeparator();
  std::string libFile = libPath + separator + libraryName();

#ifdef _WIN32
  
  typedef DLL_DIRECTORY_COOKIE(WINAPI * PAddDllDirectory)(PCWSTR);
  HMODULE hKernel = ::GetModuleHandle("kernel32.dll");

  if (hKernel == NULL) {
    lanternLoadError(pError);
    *pError = "Get Kernel - " + *pError;
    return false;
  }
  
  PAddDllDirectory add_dll_directory = (PAddDllDirectory)::GetProcAddress(hKernel, "AddDllDirectory");
  
  if (add_dll_directory != NULL) {
    std::wstring libPathWStr = std::wstring(libPath.begin(), libPath.end());
    DLL_DIRECTORY_COOKIE cookie = add_dll_directory(libPathWStr.c_str());

    if (cookie == NULL) {
      lanternLoadError(pError);
      *pError = "Add Dll Directory - " + *pError;
      return false;
    }
  }

  pLibrary = (void*)::LoadLibraryEx(libFile.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
  pLibrary = ::dlopen(libFile.c_str(), RTLD_NOW|RTLD_GLOBAL);
#endif
  if (pLibrary == NULL)
  {
    lanternLoadError(pError);
    *pError = libFile + " - " + *pError;
    return false;
  }
  else
  {
    return true;
  }
}

bool laternLoadSymbol(void* pLib, const std::string& name, void** ppSymbol, std::string* pError)
{
  *ppSymbol = NULL;
#ifdef _WIN32
  *ppSymbol = (void*)::GetProcAddress((HINSTANCE)pLib, name.c_str());
#else
  *ppSymbol = ::dlsym(pLib, name.c_str());
#endif
  if (*ppSymbol == NULL)
  {
    lanternLoadError(pError);
    *pError = name + " - " + *pError;
    return false;
  }
  else
  {
    return true;
  }
}

bool laternCloseLibrary(void* pLib, std::string* pError)
{
#ifdef _WIN32
  if (!::FreeLibrary((HMODULE)pLib))
#else
  if (::dlclose(pLib) != 0)
#endif
  {
    lanternLoadError(pError);
    return false;
  }
  else
  {
    return true;
  }
}

bool lanternInit(const std::string& libPath, std::string* pError)
{
  if (!lanternLoadLibrary(libPath, pError))
    return false;
  
  LOAD_SYMBOL(lanternTest);
  
  return true;
}

#endif
#endif