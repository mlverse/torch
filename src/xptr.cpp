// Vendored code from https://github.com/randy3k/xptr/blob/master/src/xptr.c
// MIT Licensed
// YEAR: 2017
// COPYRIGHT HOLDER: Randy Lai

#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <stdio.h>

void check_is_xptr(SEXP s) {
  if (TYPEOF(s) != EXTPTRSXP) {
    error("expect an externalptr");
  }
}

void* str2ptr(SEXP p) {
  if (!isString(p)) error("expect a string of pointer address");
  return (void*) strtol(CHAR(STRING_PTR(p)[0]), NULL, 0);
}

// [[Rcpp::export]]
SEXP xptr_address(SEXP s) {
  check_is_xptr(s);
  char* buf[20];
  sprintf((char*) buf, "%p", R_ExternalPtrAddr(s));
  return Rf_mkString((char*) buf);
}

// [[Rcpp::export]]
SEXP set_xptr_address(SEXP s, SEXP p) {
  check_is_xptr(s);
  R_SetExternalPtrAddr(s, str2ptr(p));
  return R_NilValue;
}

// [[Rcpp::export]]
SEXP set_xptr_protected(SEXP s, SEXP pro) {
  check_is_xptr(s);
  R_SetExternalPtrProtected(s, pro);
  return R_NilValue;
}


