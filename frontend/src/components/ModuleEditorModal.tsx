import type { FormEvent } from "react";
import type { ModuleEditorState } from "@/types/pipeline";

type ModuleEditorModalProps = {
  state: ModuleEditorState;
  error: string | null;
  onClose: () => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onNameChange: (value: string) => void;
  onAddAttribute: () => void;
  onAttributeChange: (
    rowId: string,
    field: "key" | "value",
    value: string
  ) => void;
  onRemoveAttribute: (rowId: string) => void;
  onDelete?: () => void;
};

export function ModuleEditorModal({
  state,
  error,
  onClose,
  onSubmit,
  onNameChange,
  onAddAttribute,
  onAttributeChange,
  onRemoveAttribute,
  onDelete,
}: ModuleEditorModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 px-6 py-12 backdrop-blur">
      <div className="w-full max-w-xl rounded-3xl border border-white/10 bg-slate-900 p-6 shadow-2xl">
        <form onSubmit={onSubmit} className="space-y-6">
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-cyan-300">
              {state.mode === "create" ? "Create module" : "Edit module"}
            </p>
            <h2 className="text-2xl font-semibold text-white">
              {state.mode === "create"
                ? "New pipeline module"
                : "Module configuration"}
            </h2>
            <p className="text-sm text-slate-400">
              Define the module identifier and its attribute payload. Values
              accept booleans, numbers, or JSON fragments.
            </p>
          </div>

          <div className="space-y-3">
            <label className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">
              Module name
            </label>
            <input
              autoFocus
              value={state.draftName}
              onChange={(event) => onNameChange(event.target.value)}
              className="w-full rounded-xl border border-white/10 bg-slate-950/70 px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-500/40"
              placeholder="e.g. signal_detect"
            />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">
                Attributes
              </label>
              <button
                type="button"
                onClick={onAddAttribute}
                className="inline-flex items-center gap-2 rounded-xl border border-cyan-400/40 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-200 transition hover:border-cyan-300/60 hover:bg-cyan-500/20"
              >
                <span className="text-base leading-none">+</span>
                Add
              </button>
            </div>
            <div className="flex flex-col gap-3">
              {state.draftAttributes.map((row) => (
                <div
                  key={row.id}
                  className="flex flex-col gap-2 rounded-2xl border border-white/5 bg-white/5 p-4 text-sm text-slate-100 sm:flex-row sm:items-center sm:gap-3"
                >
                  <div className="flex w-max flex-1 flex-col gap-2 sm:max-w-[40%]">
                    <span className="text-[10px] font-semibold uppercase tracking-[0.3em] text-slate-400">
                      Key
                    </span>
                    <input
                      value={row.key}
                      onChange={(event) =>
                        onAttributeChange(row.id, "key", event.target.value)
                      }
                      placeholder="threshold"
                      className="w-full rounded-lg border border-white/10 bg-slate-950/70 px-3 py-2 text-xs text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-500/30"
                    />
                  </div>
                  <div className="flex w-max flex-1 flex-col gap-2">
                    <span className="text-[10px] font-semibold uppercase tracking-[0.3em] text-slate-400">
                      Value
                    </span>
                    <input
                      value={row.value}
                      onChange={(event) =>
                        onAttributeChange(row.id, "value", event.target.value)
                      }
                      placeholder="0.65"
                      className="w-full rounded-lg border border-white/10 bg-slate-950/70 px-3 py-2 text-xs text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-500/30"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => onRemoveAttribute(row.id)}
                    className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.3em] text-slate-300 transition hover:border-red-400/50 hover:bg-red-500/10 hover:text-red-200"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>

          {error && (
            <p className="rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
              {error}
            </p>
          )}

          <div className="flex flex-col gap-3 pt-2 sm:flex-row sm:items-center sm:justify-between">
            {state.mode === "edit" && onDelete && (
              <button
                type="button"
                onClick={onDelete}
                className="inline-flex items-center justify-center rounded-xl border border-red-400/40 bg-red-500/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-red-200 transition hover:border-red-400/70 hover:bg-red-500/20"
              >
                Delete module
              </button>
            )}
            <div className="flex items-center justify-end gap-3 sm:ml-auto">
              <button
                type="button"
                onClick={onClose}
                className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-slate-200 transition hover:border-white/30 hover:bg-white/10"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="rounded-xl border border-cyan-400/40 bg-cyan-500/20 px-4 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-cyan-100 transition hover:border-cyan-300/60 hover:bg-cyan-500/30"
              >
                Save module
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
