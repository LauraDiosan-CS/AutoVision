export type PipelineModule = {
  id: string;
  name: string;
  config: Record<string, unknown>;
  accent: string;
};

export type PipelineLane = {
  id: string;
  label: string;
  modules: PipelineModule[];
};

export type DragPosition = {
  laneIndex: number;
  moduleIndex: number;
};

export type ModuleEditorAttributeRow = {
  id: string;
  key: string;
  value: string;
};

export type ModuleEditorState = {
  laneIndex: number;
  moduleIndex: number;
  mode: "create" | "edit";
  draftName: string;
  draftAttributes: ModuleEditorAttributeRow[];
};
