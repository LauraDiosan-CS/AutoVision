"use client";

import {
  Fragment,
  useCallback,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
} from "react";

type PipelineModule = {
  id: string;
  name: string;
  config: Record<string, unknown>;
  accent: string;
};

type PipelineLane = {
  id: string;
  label: string;
  modules: PipelineModule[];
};

type DragPosition = {
  laneIndex: number;
  moduleIndex: number;
};

type ModuleEditorAttributeRow = {
  id: string;
  key: string;
  value: string;
};

type ModuleEditorState = {
  laneIndex: number;
  moduleIndex: number;
  mode: "create" | "edit";
  draftName: string;
  draftAttributes: ModuleEditorAttributeRow[];
};

const accentPalette = [
  "#38bdf8",
  "#34d399",
  "#f472b6",
  "#f97316",
  "#c084fc",
  "#fb7185",
];

const formatValue = (value: unknown) => {
  if (value === null) return "null";
  if (value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value, null, 0);
  if (typeof value === "string") return value;
  return String(value);
};

const stringifyAttributeValue = (value: unknown) => {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
};

const inferAttributeValue = (value: string): unknown => {
  const trimmed = value.trim();
  if (trimmed === "") return "";
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  const numeric = Number(trimmed);
  if (!Number.isNaN(numeric) && trimmed !== "") return numeric;
  try {
    return JSON.parse(trimmed);
  } catch {
    return value;
  }
};

const createAttributeRow = (
  key = "",
  value = ""
): ModuleEditorAttributeRow => ({
  id: `attr-${Math.random().toString(36).slice(2, 9)}`,
  key,
  value,
});

const parsePipelines = (raw: unknown): PipelineLane[] => {
  if (!Array.isArray(raw)) {
    throw new Error("The imported JSON must be an array of pipeline lanes.");
  }

  return raw.map((lane, laneIndex) => {
    if (!lane || typeof lane !== "object" || Array.isArray(lane)) {
      throw new Error(`Lane ${laneIndex + 1} must be an object.`);
    }

    const entries = Object.entries(lane as Record<string, unknown>);

    if (entries.length === 0) {
      throw new Error(`Lane ${laneIndex + 1} cannot be empty.`);
    }

    const modules = entries.map(([name, payload], moduleIndex) => {
      const config =
        payload && typeof payload === "object" && !Array.isArray(payload)
          ? (payload as Record<string, unknown>)
          : { value: payload };

      return {
        id: `lane-${laneIndex}-module-${moduleIndex}-${name}`,
        name,
        config,
        accent: accentPalette[moduleIndex % accentPalette.length],
      } satisfies PipelineModule;
    });

    const roiCandidate = (lane as Record<string, unknown>).roi;
    let label = `Lane ${laneIndex + 1}`;

    if (
      roiCandidate &&
      typeof roiCandidate === "object" &&
      !Array.isArray(roiCandidate)
    ) {
      const roiType = (roiCandidate as { roi_type?: unknown }).roi_type;
      if (roiType) {
        label = String(roiType);
      }
    }

    return {
      id: `lane-${laneIndex}`,
      label,
      modules,
    } satisfies PipelineLane;
  });
};

const serializePipelines = (pipelines: PipelineLane[]) =>
  pipelines.map((lane) => {
    const lanePayload: Record<string, unknown> = {};
    lane.modules.forEach((module) => {
      lanePayload[module.name] = module.config;
    });
    return lanePayload;
  });

export default function Home() {
  const [pipelines, setPipelines] = useState<PipelineLane[]>([]);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [activeFileName, setActiveFileName] = useState<string>("");
  const [isDragActive, setIsDragActive] = useState(false);
  const dragOriginRef = useRef<DragPosition | null>(null);
  const [editorState, setEditorState] = useState<ModuleEditorState | null>(
    null
  );
  const [editorError, setEditorError] = useState<string | null>(null);

  const resetWorkspace = useCallback(() => {
    setPipelines([]);
    setWorkspaceError(null);
    setActiveFileName("");
  }, []);

  const importJson = useCallback(
    async (file: File) => {
      try {
        const text = await file.text();
        const parsed = JSON.parse(text);
        const nextPipelines = parsePipelines(parsed);
        setPipelines(nextPipelines);
        setWorkspaceError(null);
        setActiveFileName(file.name);
      } catch (error) {
        console.error(error);
        resetWorkspace();
        setWorkspaceError(
          error instanceof Error
            ? error.message
            : "Unable to read the selected file."
        );
      }
    },
    [resetWorkspace]
  );

  const handleFileSelection = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      await importJson(file);
    },
    [importJson]
  );

  const handleDropZoneDragOver = useCallback(
    (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      setIsDragActive(true);
    },
    []
  );

  const handleDropZoneDragLeave = useCallback(() => {
    setIsDragActive(false);
  }, []);

  const handleDropZoneFileDrop = useCallback(
    async (event: DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      setIsDragActive(false);
      const file = event.dataTransfer.files?.[0];
      if (!file) return;
      await importJson(file);
    },
    [importJson]
  );

  const handleDragStart = useCallback(
    (laneIndex: number, moduleIndex: number) => {
      dragOriginRef.current = { laneIndex, moduleIndex };
    },
    []
  );

  const handleDragEnd = useCallback(() => {
    dragOriginRef.current = null;
  }, []);

  const handleModuleDrop = useCallback(
    (targetLaneIndex: number, targetModuleIndex: number) => {
      const origin = dragOriginRef.current;
      dragOriginRef.current = null;
      if (!origin) return;

      setPipelines((previous) => {
        const draft = previous.map((lane) => ({
          ...lane,
          modules: [...lane.modules],
        }));
        const sourceLane = draft[origin.laneIndex];
        if (!sourceLane) return previous;

        const [module] = sourceLane.modules.splice(origin.moduleIndex, 1);
        if (!module) return previous;

        const destinationLane = draft[targetLaneIndex];
        if (!destinationLane) return previous;

        let insertionIndex = targetModuleIndex;
        if (
          origin.laneIndex === targetLaneIndex &&
          origin.moduleIndex < targetModuleIndex
        ) {
          insertionIndex -= 1;
        }

        destinationLane.modules.splice(
          Math.max(0, Math.min(insertionIndex, destinationLane.modules.length)),
          0,
          module
        );
        return draft;
      });
    },
    []
  );

  const allowDrop = useCallback((event: DragEvent) => {
    event.preventDefault();
  }, []);

  const closeEditor = useCallback(() => {
    setEditorState(null);
    setEditorError(null);
  }, []);

  const handleExport = useCallback(() => {
    if (pipelines.length === 0) {
      return;
    }

    const serialized = serializePipelines(pipelines);
    const blob = new Blob([JSON.stringify(serialized, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const suggestedName = activeFileName
      ? `${activeFileName.replace(/\.json$/i, "")}-export.json`
      : "pipelines-export.json";
    anchor.href = url;
    anchor.download = suggestedName;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [activeFileName, pipelines]);

  const openCreateModule = useCallback(
    (laneIndex: number, moduleIndex: number) => {
      setEditorError(null);
      setEditorState({
        laneIndex,
        moduleIndex,
        mode: "create",
        draftName: "",
        draftAttributes: [createAttributeRow()],
      });
    },
    []
  );

  const openEditModule = useCallback(
    (laneIndex: number, moduleIndex: number) => {
      const lane = pipelines[laneIndex];
      const selectedModule = lane?.modules[moduleIndex];
      if (!selectedModule) return;

      const attributeEntries = Object.entries(selectedModule.config);

      setEditorError(null);
      setEditorState({
        laneIndex,
        moduleIndex,
        mode: "edit",
        draftName: selectedModule.name,
        draftAttributes:
          attributeEntries.length > 0
            ? attributeEntries.map(([key, value]) =>
                createAttributeRow(key, stringifyAttributeValue(value))
              )
            : [createAttributeRow()],
      });
    },
    [pipelines]
  );

  const handleEditorNameChange = useCallback((value: string) => {
    setEditorError(null);
    setEditorState((previous) => {
      if (!previous) return previous;
      return { ...previous, draftName: value };
    });
  }, []);

  const handleAttributeChange = useCallback(
    (rowId: string, field: "key" | "value", value: string) => {
      setEditorError(null);
      setEditorState((previous) => {
        if (!previous) return previous;
        return {
          ...previous,
          draftAttributes: previous.draftAttributes.map((row) =>
            row.id === rowId
              ? {
                  ...row,
                  [field]: value,
                }
              : row
          ),
        };
      });
    },
    []
  );

  const handleAddAttributeRow = useCallback(() => {
    setEditorError(null);
    setEditorState((previous) => {
      if (!previous) return previous;
      return {
        ...previous,
        draftAttributes: [...previous.draftAttributes, createAttributeRow()],
      };
    });
  }, []);

  const handleRemoveAttributeRow = useCallback((rowId: string) => {
    setEditorError(null);
    setEditorState((previous) => {
      if (!previous) return previous;
      const remaining = previous.draftAttributes.filter(
        (row) => row.id !== rowId
      );
      return {
        ...previous,
        draftAttributes:
          remaining.length > 0 ? remaining : [createAttributeRow()],
      };
    });
  }, []);

  const handleEditorSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!editorState) return;

      const name = editorState.draftName.trim();
      if (!name) {
        setEditorError("Module name is required.");
        return;
      }

      const sanitizedAttributes = editorState.draftAttributes
        .map(({ key, value }) => ({ key: key.trim(), value: value.trim() }))
        .filter(({ key }) => key !== "");

      const config: Record<string, unknown> = {};
      sanitizedAttributes.forEach(({ key, value }) => {
        config[key] = inferAttributeValue(value);
      });

      setPipelines((previous) => {
        const draft = previous.map((lane) => ({
          ...lane,
          modules: [...lane.modules],
        }));
        const lane = draft[editorState.laneIndex];
        if (!lane) return previous;

        if (editorState.mode === "create") {
          const accent =
            accentPalette[Math.floor(Math.random() * accentPalette.length)];
          const newModule: PipelineModule = {
            id: `module-${Math.random().toString(36).slice(2, 10)}`,
            name,
            config,
            accent,
          };
          const insertionIndex = Math.max(
            0,
            Math.min(editorState.moduleIndex, lane.modules.length)
          );
          lane.modules.splice(insertionIndex, 0, newModule);
        } else {
          const target = lane.modules[editorState.moduleIndex];
          if (!target) return previous;
          target.name = name;
          target.config = config;
        }

        return draft;
      });

      closeEditor();
    },
    [closeEditor, editorState]
  );

  const handleEditorDelete = useCallback(() => {
    if (!editorState || editorState.mode !== "edit") return;

    setPipelines((previous) => {
      const draft = previous.map((lane) => ({
        ...lane,
        modules: [...lane.modules],
      }));
      const lane = draft[editorState.laneIndex];
      if (!lane) return previous;
      lane.modules.splice(editorState.moduleIndex, 1);
      return draft;
    });

    closeEditor();
  }, [closeEditor, editorState]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <main className="mx-auto flex w-full max-w-11/12 flex-col gap-10 px-6 py-12">
        <header className="rounded-3xl border border-white/10 bg-white/[0.03] p-8 shadow-[0_20px_80px_-40px_rgba(15,118,110,0.75)] backdrop-blur">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="space-y-3">
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-300">
                Pipeline Composer
              </p>
              <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                Import, visualize, and refactor perception pipelines.
              </h1>
              <p className="text-sm text-slate-300">
                Load a JSON configuration to project each processing lane. Drag
                modules to reorder or re-route between lanes and instantly
                explore alternative topologies.
              </p>
            </div>
          </div>

          <div className="mt-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-center">
            <label
              onDragOver={handleDropZoneDragOver}
              onDragEnter={handleDropZoneDragOver}
              onDragLeave={handleDropZoneDragLeave}
              onDrop={handleDropZoneFileDrop}
              className={`group relative flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-4 py-6 transition ${
                isDragActive
                  ? "border-cyan-400 bg-cyan-400/10"
                  : "border-white/10 bg-white/5 hover:border-cyan-400/60"
              }`}
            >
              <input
                type="file"
                accept="application/json"
                onChange={handleFileSelection}
                className="hidden"
              />
              <div className="flex size-12 items-center justify-center rounded-full bg-cyan-400/10 text-cyan-300">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  className="size-6"
                >
                  <path d="M12 3a1 1 0 0 1 1 1v9.586l2.293-2.293a1 1 0 1 1 1.414 1.414l-4 4a1 1 0 0 1-1.414 0l-4-4a1 1 0 1 1 1.414-1.414L11 13.586V4a1 1 0 0 1 1-1Z" />
                  <path d="M5 15a1 1 0 0 1 2 0v3h10v-3a1 1 0 1 1 2 0v4a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1v-4Z" />
                </svg>
              </div>
              <div className="text-center">
                <p className="font-medium text-slate-100">
                  Drop a JSON file here or click to browse
                </p>
              </div>
              {activeFileName && (
                <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-xs text-cyan-200">
                  {activeFileName}
                </span>
              )}
            </label>
          </div>

          {workspaceError && (
            <p className="mt-4 rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
              {workspaceError}
            </p>
          )}
        </header>

        <section className="flex flex-col gap-6">
          {/* <div className="flex flex-col gap-2">
            <h2 className="text-xl font-semibold text-white">
              Pipeline canvas
            </h2>
            <p className="text-sm text-slate-400">
              Arrange and connect modules within each lane. Dragging a card into
              another lane reassigns its flow. Drop indicators appear as you
              hover over a lane or between cards.
            </p>
          </div> */}

          <div className="overflow-x-auto">
            <div className="inline-block min-w-full w-fit rounded-3xl border border-white/10 bg-slate-900/40 p-6 shadow-inner">
              <div className="space-y-3 mb-5">
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-300">
                  Pipeline canvas
                </p>
                <p className="text-sm text-slate-300">
                  Arrange and connect modules within each lane. Dragging a card
                  into another lane reassigns its flow. Drop indicators appear
                  as you hover over a lane or between cards.
                </p>
              </div>

              {pipelines.length === 0 ? (
                <div className="flex min-h-[280px] items-center justify-center text-sm text-slate-500">
                  Import a pipeline json file to begin composing.
                </div>
              ) : (
                <div className="grid gap-6 md:grid-cols-12">
                  <div className="relative flex flex-col items-center justify-center gap-4 rounded-2xl border border-white/10 bg-white/5 px-6 py-10 text-center shadow-lg md:col-span-2">
                    <span className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-300">
                      Source
                    </span>
                    <h3 className="text-lg font-semibold text-white">
                      Data Stream
                    </h3>
                    <p className="text-xs text-slate-400">
                      Origin input feeding every perception lane.
                    </p>
                  </div>

                  <div className="flex flex-col gap-6 md:col-span-8">
                    {pipelines.map((lane, laneIndex) => (
                      <div key={lane.id} className="relative">
                        <div className="pointer-events-none absolute left-[-48px] top-1/2 hidden h-px w-12 -translate-y-1/2 bg-gradient-to-r from-cyan-500/0 via-cyan-300 to-cyan-200 md:block" />
                        <div className="pointer-events-none absolute right-[-48px] top-1/2 hidden h-px w-12 -translate-y-1/2 bg-gradient-to-r from-blue-400 to-blue-400/0 md:block" />
                        <div className="rounded-3xl border border-white/10 bg-white/[0.04] shadow-lg backdrop-blur">
                          <div className="flex flex-wrap gap-4 p-4 md:p-6">
                            {Array.from({
                              length: lane.modules.length + 1,
                            }).map((_, insertionIndex) => {
                              const moduleCard = lane.modules[insertionIndex];
                              const forceVisibleButton =
                                lane.modules.length === 0;

                              return (
                                <Fragment
                                  key={`${lane.id}-insert-${insertionIndex}`}
                                >
                                  <div
                                    className="group relative flex min-w-[40px] items-center justify-center"
                                    onDragOver={allowDrop}
                                    onDrop={(event) => {
                                      event.preventDefault();
                                      handleModuleDrop(
                                        laneIndex,
                                        insertionIndex
                                      );
                                    }}
                                  >
                                    <button
                                      type="button"
                                      aria-label="Insert module"
                                      onClick={() =>
                                        openCreateModule(
                                          laneIndex,
                                          insertionIndex
                                        )
                                      }
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
                                    <div className="flex flex-col justify-center">
                                      <div
                                        role="button"
                                        tabIndex={0}
                                        draggable
                                        onDragStart={() =>
                                          handleDragStart(
                                            laneIndex,
                                            insertionIndex
                                          )
                                        }
                                        onDragEnd={handleDragEnd}
                                        onDragOver={allowDrop}
                                        onDrop={(event) => {
                                          event.preventDefault();
                                          handleModuleDrop(
                                            laneIndex,
                                            insertionIndex
                                          );
                                        }}
                                        onClick={() =>
                                          openEditModule(
                                            laneIndex,
                                            insertionIndex
                                          )
                                        }
                                        onKeyDown={(event) => {
                                          if (
                                            event.key === "Enter" ||
                                            event.key === " "
                                          ) {
                                            event.preventDefault();
                                            openEditModule(
                                              laneIndex,
                                              insertionIndex
                                            );
                                          }
                                        }}
                                        className="min-w-[200px] rounded-2xl border bg-slate-950/60 p-3 text-left outline-none transition hover:-translate-y-1 hover:shadow-[0_20px_45px_-24px_rgba(56,189,248,0.65)] focus:-translate-y-1 focus:shadow-[0_20px_45px_-24px_rgba(56,189,248,0.65)]"
                                        style={{
                                          borderColor: `${moduleCard.accent}33`,
                                          boxShadow: `0 10px 30px -20px ${moduleCard.accent}80`,
                                        }}
                                      >
                                        <div className="flex items-center justify-between gap-3">
                                          <div className="flex items-center gap-2">
                                            <span
                                              className="size-2 rounded-full"
                                              style={{
                                                backgroundColor:
                                                  moduleCard.accent,
                                              }}
                                            />
                                            <p className="text-sm font-semibold text-white">
                                              {moduleCard.name}
                                            </p>
                                          </div>
                                          <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                                            {`#${insertionIndex + 1}`}
                                          </span>
                                        </div>

                                        <div className="mt-3 flex flex-wrap gap-2">
                                          {Object.entries(
                                            moduleCard.config
                                          ).map(([key, value]) => (
                                            <span
                                              key={key}
                                              className="rounded-full bg-white/5 px-3 py-1 text-[11px] text-slate-200"
                                            >
                                              <span className="font-medium text-slate-300">
                                                {key}:
                                              </span>{" "}
                                              {formatValue(value)}
                                            </span>
                                          ))}
                                          {Object.keys(moduleCard.config)
                                            .length === 0 && (
                                            <span className="rounded-full bg-white/5 px-3 py-1 text-[11px] text-slate-400">
                                              No attributes
                                            </span>
                                          )}
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                </Fragment>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="relative flex flex-col items-center justify-center gap-4 rounded-2xl border border-white/10 bg-white/5 px-6 py-10 text-center shadow-lg md:col-span-2">
                    <span className="text-sm font-semibold uppercase tracking-[0.2em] text-blue-300">
                      Output
                    </span>
                    <h3 className="text-lg font-semibold text-white">
                      Data Merger
                    </h3>
                    <p className="text-xs text-slate-400">
                      Unified output after routing each perception stream.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        <footer className="mt-10">
          <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-6 shadow-[0_20px_60px_-40px_rgba(56,189,248,0.65)] backdrop-blur">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div className="space-y-1 text-sm text-slate-300">
                <p className="text-xs font-semibold uppercase tracking-[0.25em] text-cyan-300">
                  Export json
                </p>
                <p>Download the current pipeline as a JSON file.</p>
              </div>
              <button
                type="button"
                onClick={handleExport}
                disabled={pipelines.length === 0}
                className={`inline-flex items-center justify-center rounded-xl border px-5 py-3 text-sm font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300 ${
                  pipelines.length === 0
                    ? "cursor-not-allowed border-white/10 bg-white/5 text-slate-500"
                    : "border-cyan-400/40 bg-cyan-500/20 text-cyan-100 hover:border-cyan-300/60 hover:bg-cyan-500/30"
                }`}
              >
                Export JSON
              </button>
            </div>
          </div>
        </footer>
      </main>

      {editorState && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 px-6 py-12 backdrop-blur">
          <div className="w-full max-w-xl rounded-3xl border border-white/10 bg-slate-900 p-6 shadow-2xl">
            <form onSubmit={handleEditorSubmit} className="space-y-6">
              <div className="space-y-2">
                <p className="text-xs font-semibold uppercase tracking-[0.3em] text-cyan-300">
                  {editorState.mode === "create"
                    ? "Create module"
                    : "Edit module"}
                </p>
                <h2 className="text-2xl font-semibold text-white">
                  {editorState.mode === "create"
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
                  value={editorState.draftName}
                  onChange={(event) =>
                    handleEditorNameChange(event.target.value)
                  }
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
                    onClick={handleAddAttributeRow}
                    className="inline-flex items-center gap-2 rounded-xl border border-cyan-400/40 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-200 transition hover:border-cyan-300/60 hover:bg-cyan-500/20"
                  >
                    <span className="text-base leading-none">+</span>
                    Add
                  </button>
                </div>
                <div className="flex flex-col gap-3">
                  {editorState.draftAttributes.map((row) => (
                    <div
                      key={row.id}
                      className="flex flex-col gap-2 rounded-2xl border border-white/5 bg-white/5 p-4 text-sm text-slate-100 sm:flex-row sm:items-center sm:gap-3"
                    >
                      <div className="flex w-full flex-1 flex-col gap-2 sm:max-w-[40%]">
                        <span className="text-[10px] font-semibold uppercase tracking-[0.3em] text-slate-400">
                          Key
                        </span>
                        <input
                          value={row.key}
                          onChange={(event) =>
                            handleAttributeChange(
                              row.id,
                              "key",
                              event.target.value
                            )
                          }
                          placeholder="threshold"
                          className="w-full rounded-lg border border-white/10 bg-slate-950/70 px-3 py-2 text-xs text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-500/30"
                        />
                      </div>
                      <div className="flex w-full flex-1 flex-col gap-2">
                        <span className="text-[10px] font-semibold uppercase tracking-[0.3em] text-slate-400">
                          Value
                        </span>
                        <input
                          value={row.value}
                          onChange={(event) =>
                            handleAttributeChange(
                              row.id,
                              "value",
                              event.target.value
                            )
                          }
                          placeholder="0.65"
                          className="w-full rounded-lg border border-white/10 bg-slate-950/70 px-3 py-2 text-xs text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-500/30"
                        />
                      </div>
                      <button
                        type="button"
                        onClick={() => handleRemoveAttributeRow(row.id)}
                        className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.3em] text-slate-300 transition hover:border-red-400/50 hover:bg-red-500/10 hover:text-red-200"
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {editorError && (
                <p className="rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                  {editorError}
                </p>
              )}

              <div className="flex flex-col gap-3 pt-2 sm:flex-row sm:items-center sm:justify-between">
                {editorState.mode === "edit" && (
                  <button
                    type="button"
                    onClick={handleEditorDelete}
                    className="inline-flex items-center justify-center rounded-xl border border-red-400/40 bg-red-500/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-red-200 transition hover:border-red-400/70 hover:bg-red-500/20"
                  >
                    Delete module
                  </button>
                )}
                <div className="flex items-center justify-end gap-3 sm:ml-auto">
                  <button
                    type="button"
                    onClick={closeEditor}
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
      )}
    </div>
  );
}
