import { Fragment, useEffect, useRef, useState } from "react";
import type { PipelineLane as PipelineLaneType } from "@/types/pipeline";
import type { DragEvent, FormEvent } from "react";
import { ModuleCard } from "./ModuleCard";

type PipelineLaneProps = {
  lane: PipelineLaneType;
  laneIndex: number;
  onAllowDrop: (event: DragEvent<HTMLElement>) => void;
  onModuleDrop: (laneIndex: number, moduleIndex: number) => void;
  onCreateModule: (laneIndex: number, moduleIndex: number) => void;
  onEditModule: (laneIndex: number, moduleIndex: number) => void;
  onDragStart: (laneIndex: number, moduleIndex: number) => void;
  onDragEnd: () => void;
  onRenameLane: (laneIndex: number, label: string) => void;
  onDeleteLane: (laneIndex: number) => void;
};

export function PipelineLane({
  lane,
  laneIndex,
  onAllowDrop,
  onModuleDrop,
  onCreateModule,
  onEditModule,
  onDragStart,
  onDragEnd,
  onRenameLane,
  onDeleteLane,
}: PipelineLaneProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isEditingLabel, setIsEditingLabel] = useState(false);
  const [draftLabel, setDraftLabel] = useState(lane.label);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    setDraftLabel(lane.label);
  }, [lane.label]);

  useEffect(() => {
    if (!isMenuOpen) return;

    const handleClickAway = (event: MouseEvent) => {
      if (!menuRef.current) return;
      if (menuRef.current.contains(event.target as Node)) return;

      setIsMenuOpen(false);
      setIsEditingLabel(false);
      setDraftLabel(lane.label);
    };

    document.addEventListener("mousedown", handleClickAway);
    return () => {
      document.removeEventListener("mousedown", handleClickAway);
    };
  }, [isMenuOpen, lane.label]);

  useEffect(() => {
    if (isMenuOpen && isEditingLabel && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isMenuOpen, isEditingLabel]);

  const handleRenameSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = draftLabel.trim();
    if (!trimmed) return;

    onRenameLane(laneIndex, trimmed);
    setIsEditingLabel(false);
    setIsMenuOpen(false);
  };

  return (
    <div className="relative">
      <div className="pointer-events-none absolute left-[-48px] top-1/2 hidden h-px w-12 -translate-y-1/2 bg-gradient-to-r from-cyan-500/0 via-cyan-300 to-cyan-200 md:block" />
      <div className="pointer-events-none absolute right-[-48px] top-1/2 hidden h-px w-12 -translate-y-1/2 bg-gradient-to-r from-blue-400 to-blue-400/0 md:block" />
      <div className="rounded-3xl border border-white/10 bg-white/[0.04] shadow-lg backdrop-blur">
        <div className="flex items-start justify-between gap-3 border-b border-white/10 px-6 py-4">
          <div>
            <h3 className="text-md font-semibold text-white">
              <span className="text-lg font-semibold uppercase tracking-[0.2em] text-cyan-300 mr-2">
                Lane:
              </span>
              {lane.label}
            </h3>
          </div>
          <div ref={menuRef} className="relative ml-auto">
            <button
              type="button"
              aria-haspopup="menu"
              aria-expanded={isMenuOpen}
              onClick={() => {
                setIsMenuOpen((previous) => !previous);
                setIsEditingLabel(false);
                setDraftLabel(lane.label);
              }}
              className="flex size-8 items-center justify-center rounded-full border border-white/10 bg-slate-900/80 text-lg font-semibold text-slate-200 transition hover:border-cyan-400/60 hover:bg-cyan-500/10 hover:text-cyan-100 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 focus:ring-offset-slate-900"
            >
              ...
            </button>
            {isMenuOpen && (
              <div className="absolute right-0 z-10 mt-2 w-56 rounded-2xl border border-white/10 bg-slate-950/95 p-4 text-left shadow-xl">
                {isEditingLabel ? (
                  <form onSubmit={handleRenameSubmit} className="space-y-3">
                    <div className="space-y-1">
                      <label
                        htmlFor={`${lane.id}-rename`}
                        className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400"
                      >
                        Lane name
                      </label>
                      <input
                        id={`${lane.id}-rename`}
                        ref={inputRef}
                        type="text"
                        value={draftLabel}
                        onChange={(event) => setDraftLabel(event.target.value)}
                        className="w-full rounded-xl border border-white/10 bg-slate-900 px-3 py-2 text-sm text-white outline-none transition focus:border-cyan-300 focus:ring-2 focus:ring-cyan-400"
                        placeholder="Enter lane name"
                      />
                    </div>
                    <div className="flex justify-end gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          setIsEditingLabel(false);
                          setDraftLabel(lane.label);
                        }}
                        className="rounded-full border border-white/10 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.2em] text-slate-300 transition hover:border-slate-400 hover:text-white"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        disabled={!draftLabel.trim()}
                        className="rounded-full border border-cyan-400/60 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-200 transition hover:border-cyan-300 hover:bg-cyan-500/20 hover:text-cyan-100 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        Save
                      </button>
                    </div>
                  </form>
                ) : (
                  <div className="flex flex-col gap-2 text-sm">
                    <button
                      type="button"
                      onClick={() => setIsEditingLabel(true)}
                      className="rounded-xl px-3 py-2 text-left text-slate-200 transition hover:bg-cyan-500/10 hover:text-cyan-100"
                    >
                      Rename lane
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        onDeleteLane(laneIndex);
                        setIsMenuOpen(false);
                      }}
                      className="rounded-xl px-3 py-2 text-left text-red-300 transition hover:bg-red-500/10 hover:text-red-200"
                    >
                      Delete lane
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-wrap gap-4 p-4 md:p-6">
          {Array.from({ length: lane.modules.length + 1 }).map(
            (_, insertionIndex) => {
              const moduleCard = lane.modules[insertionIndex];
              const forceVisibleButton = lane.modules.length === 0;

              return (
                <Fragment key={`${lane.id}-insert-${insertionIndex}`}>
                  <div
                    className="group relative flex min-w-[40px] items-center justify-center"
                    onDragOver={onAllowDrop}
                    onDrop={(event) => {
                      event.preventDefault();
                      onModuleDrop(laneIndex, insertionIndex);
                    }}
                  >
                    <button
                      type="button"
                      aria-label="Insert module"
                      onClick={() => onCreateModule(laneIndex, insertionIndex)}
                      className={`flex size-8 items-center justify-center rounded-full border border-dashed border-cyan-400/40 bg-slate-950/80 text-lg font-semibold text-cyan-200 transition hover:border-cyan-300 hover:bg-cyan-500/10 hover:text-cyan-100 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 focus:ring-offset-slate-900 ${
                        forceVisibleButton
                          ? "opacity-100"
                          : "opacity-0 group-hover:opacity-100"
                      }`}
                    >
                      +
                    </button>
                  </div>

                  {moduleCard && (
                    <ModuleCard
                      moduleCard={moduleCard}
                      laneIndex={laneIndex}
                      moduleIndex={insertionIndex}
                      onDragStart={onDragStart}
                      onDragEnd={onDragEnd}
                      onAllowDrop={onAllowDrop}
                      onDrop={onModuleDrop}
                      onOpenEditor={onEditModule}
                    />
                  )}
                </Fragment>
              );
            }
          )}
        </div>
      </div>
    </div>
  );
}
