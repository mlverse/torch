#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <torch/script.h> // One-stop header.
#include "utils.hpp"

void* _lantern_jit_script_create_Decl (void* range, void* params, void* return_type)
{
    auto out = torch::jit::Decl::create(
        *reinterpret_cast<torch::jit::SourceRange*>(range),
        *reinterpret_cast<torch::jit::List<torch::jit::Param>*>(params),
        *reinterpret_cast<torch::jit::Maybe<torch::jit::Expr>*>(return_type)
    );
    return (void*) new torch::jit::Decl(out);
}

void* _lantern_jit_script_create_SourceRange(void* source, int start, int end)
{
    auto source_ = std::shared_ptr<torch::jit::Source>(
        reinterpret_cast<torch::jit::Source*>(source), 
        [](void*x) {}
    );

    auto out = torch::jit::SourceRange(source_, start, end);

    return (void*) new torch::jit::SourceRange(out);
}

void* _lantern_jit_script_create_Source (void* text)
{
    return (void*) new torch::jit::Source(
        reinterpret_cast<LanternObject<std::string>*>(text)->get()
    );
}

void* _lantern_jit_script_create_Ident (void* range, void* name)
{
    auto out = torch::jit::Ident::create(
        *reinterpret_cast<torch::jit::SourceRange*>(range),
        reinterpret_cast<LanternObject<std::string>*>(name)->get()
    );
    return (void*) new torch::jit::Ident(out);
}

void* _lantern_jit_script_create_Param (void* range, void* ident, void* type, void* def, bool kwarg_only)
{
    auto out = torch::jit::Param::create(
        *reinterpret_cast<torch::jit::SourceRange*>(range),
        *reinterpret_cast<torch::jit::Ident*>(ident),
        *reinterpret_cast<torch::jit::Maybe<torch::jit::Expr>*>(type),
        *reinterpret_cast<torch::jit::Maybe<torch::jit::Expr>*>(def),
        kwarg_only
    );
    return (void*) new torch::jit::Param(out);
}

void* _lantern_jit_script_create_ExprStmt (void* range, void* list)
{
    auto out = torch::jit::ExprStmt::create(
        *reinterpret_cast<torch::jit::SourceRange*>(range),
        {*reinterpret_cast<torch::jit::Expr*>(list)}
    );
    return (void *) new torch::jit::ExprStmt(out);
}

void* _lantern_jit_script_create_Const (void* range, void* value)
{
    auto out = torch::jit::Const::create(
        *reinterpret_cast<torch::jit::SourceRange*>(range),
        reinterpret_cast<LanternObject<std::string>*>(value)->get()
    );
    return (void *) new torch::jit::Const(out);
}

void* _lantern_jit_script_create_Return (void* range, void* value)
{
    auto out = torch::jit::Return::create(
        *reinterpret_cast<torch::jit::SourceRange*>(range),
        *reinterpret_cast<torch::jit::Expr*>(value)
    );
    return (void *) new torch::jit::Return(out);
}