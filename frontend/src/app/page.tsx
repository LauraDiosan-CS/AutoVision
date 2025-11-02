"use client";

import { useCallback, useRef, useState, type FormEvent } from "react";

import { AppHeader } from "@/components/AppHeader";
import { PipelineCanvas } from "@/components/PipelineCanvas";
import { ExportFooter } from "@/components/ExportFooter";
import { ModuleEditorModal } from "@/components/ModuleEditorModal";
import type {
  DragPosition,
  ModuleEditorState,
  PipelineLane,
} from "@/types/pipeline";
import {
  accentPalette,
  createAttributeRow,
  inferAttributeValue,
  parsePipelines,
  serializePipelines,
  stringifyAttributeValue,
} from "@/utils/pipeline";

type ModuleEditorStateNullable = ModuleEditorState | null;
type HandleModuleDrop = (
  targetLaneIndex: number,
  targetModuleIndex: number
) => void;
type HandleDragStart = (laneIndex: number, moduleIndex: number) => void;
type HandleModuleEdit = (laneIndex: number, moduleIndex: number) => void;

export default function Home() {
  const [pipelines, setPipelines] = useState<PipelineLane[]>([]);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [activeFileName, setActiveFileName] = useState<string>("");
  const dragOriginRef = useRef<DragPosition | null>(null);
  const [editorState, setEditorState] =
    useState<ModuleEditorStateNullable>(null);
  const [editorError, setEditorError] = useState<string | null>(null);

  const createPlaceholderModule = useCallback(() => {
    const accent =
      accentPalette[Math.floor(Math.random() * accentPalette.length)];

    return {
      id: `module-${Math.random().toString(36).slice(2, 10)}`,
      name: "New Module",
      config: {
        description: "Placeholder module",
      },
      accent,
    };
  }, []);

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

  const handleDragStart = useCallback<HandleDragStart>(
    (laneIndex, moduleIndex) => {
      dragOriginRef.current = { laneIndex, moduleIndex };
    },
    []
  );

  const handleDragEnd = useCallback(() => {
    dragOriginRef.current = null;
  }, []);

  const handleModuleDrop = useCallback<HandleModuleDrop>(
    (targetLaneIndex, targetModuleIndex) => {
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
        return draft.filter((lane) => lane.modules.length > 0);
      });
    },
    []
  );

  const allowDrop = useCallback((event: React.DragEvent<HTMLElement>) => {
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

  const openEditModule = useCallback<HandleModuleEdit>(
    (laneIndex, moduleIndex) => {
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
            row.id === rowId ? { ...row, [field]: value } : row
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
          const newModule = {
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
      return draft.filter((candidate) => candidate.modules.length > 0);
    });

    closeEditor();
  }, [closeEditor, editorState]);

  const handleAddLane = useCallback(() => {
    setPipelines((previous) => {
      const newLane: PipelineLane = {
        id: `lane-${Math.random().toString(36).slice(2, 10)}`,
        label: `Lane ${previous.length + 1}`,
        modules: [createPlaceholderModule()],
      };

      return [...previous, newLane];
    });
  }, [createPlaceholderModule]);

  const handleRenameLane = useCallback((laneIndex: number, label: string) => {
    const trimmed = label.trim();
    if (!trimmed) return;

    setPipelines((previous) =>
      previous.map((lane, index) =>
        index === laneIndex ? { ...lane, label: trimmed } : lane
      )
    );
  }, []);

  const handleDeleteLane = useCallback(
    (laneIndex: number) => {
      setPipelines((previous) =>
        previous.filter((_, index) => index !== laneIndex)
      );
      closeEditor();
    },
    [closeEditor]
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <main className="mx-auto flex w-max max-w-11/12 flex-col gap-10 px-6 py-12">
        <AppHeader importJson={importJson} />
        {workspaceError && (
          <p className="mt-4 rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {workspaceError}
          </p>
        )}
        <PipelineCanvas
          pipelines={pipelines}
          onAllowDrop={allowDrop}
          onModuleDrop={handleModuleDrop}
          onCreateModule={openCreateModule}
          onEditModule={openEditModule}
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
          onAddLane={handleAddLane}
          onRenameLane={handleRenameLane}
          onDeleteLane={handleDeleteLane}
        />
        <ExportFooter
          onExport={handleExport}
          disabled={pipelines.length === 0}
        />
      </main>

      {editorState && (
        <ModuleEditorModal
          state={editorState}
          error={editorError}
          onClose={closeEditor}
          onSubmit={handleEditorSubmit}
          onNameChange={handleEditorNameChange}
          onAddAttribute={handleAddAttributeRow}
          onAttributeChange={handleAttributeChange}
          onRemoveAttribute={handleRemoveAttributeRow}
          onDelete={
            editorState.mode === "edit" ? handleEditorDelete : undefined
          }
        />
      )}
    </div>
  );
}
