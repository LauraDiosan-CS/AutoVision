import { Fragment } from "react";
import type { PipelineLane as PipelineLaneType } from "@/types/pipeline";
import type { DragEvent } from "react";
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
}: PipelineLaneProps) {
  return (
    <div className="relative">
      <div className="pointer-events-none absolute left-[-48px] top-1/2 hidden h-px w-12 -translate-y-1/2 bg-gradient-to-r from-cyan-500/0 via-cyan-300 to-cyan-200 md:block" />
      <div className="pointer-events-none absolute right-[-48px] top-1/2 hidden h-px w-12 -translate-y-1/2 bg-gradient-to-r from-blue-400 to-blue-400/0 md:block" />
      <div className="rounded-3xl border border-white/10 bg-white/[0.04] shadow-lg backdrop-blur">
        <div className="flex flex-wrap gap-4 p-4 md:p-6">
          {Array.from({ length: lane.modules.length + 1 }).map((_, insertionIndex) => {
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
                      forceVisibleButton ? "opacity-100" : "opacity-0 group-hover:opacity-100"
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
          })}
        </div>
      </div>
    </div>
  );
}
