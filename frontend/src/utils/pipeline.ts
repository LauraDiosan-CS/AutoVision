import type {
  ModuleEditorAttributeRow,
  PipelineLane,
  PipelineModule,
} from "@/types/pipeline";

export const accentPalette = [
  "#38bdf8",
  "#34d399",
  "#f472b6",
  "#f97316",
  "#c084fc",
  "#fb7185",
];

export const formatValue = (value: unknown) => {
  if (value === null) return "null";
  if (value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value, null, 0);
  if (typeof value === "string") return value;
  return String(value);
};

export const stringifyAttributeValue = (value: unknown) => {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
};

export const inferAttributeValue = (value: string): unknown => {
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

export const createAttributeRow = (
  key = "",
  value = "",
): ModuleEditorAttributeRow => ({
  id: `attr-${Math.random().toString(36).slice(2, 9)}`,
  key,
  value,
});

export const parsePipelines = (raw: unknown): PipelineLane[] => {
  if (!Array.isArray(raw)) {
    throw new Error("The imported JSON must be an array of pipeline lanes.");
  }

  return raw.map((lane, laneIndex) => {
    if (!lane || typeof lane !== "object" || Array.isArray(lane)) {
      throw new Error(`Lane ${laneIndex + 1} must be an object.`);
    }

    const laneRecord = lane as Record<string, unknown>;
    const filtersCandidate = laneRecord.filters;
    const hasFilterGroup =
      filtersCandidate &&
      typeof filtersCandidate === "object" &&
      !Array.isArray(filtersCandidate);

    const moduleSource = hasFilterGroup
      ? (filtersCandidate as Record<string, unknown>)
      : laneRecord;

    const entries = Object.entries(moduleSource);

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

    const roiCandidate = moduleSource.roi;
    const hasExplicitName =
      hasFilterGroup && typeof laneRecord.name === "string";
    let label = `Lane ${laneIndex + 1}`;

    if (hasExplicitName) {
      const trimmedName = (laneRecord.name as string).trim();
      if (trimmedName) {
        label = trimmedName;
      }
    } else if (
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

export const serializePipelines = (pipelines: PipelineLane[]) =>
  pipelines.map((lane) => {
    const filters: Record<string, unknown> = {};
    lane.modules.forEach((module) => {
      filters[module.name] = module.config;
    });

    return {
      name: lane.label,
      filters,
    };
  });
