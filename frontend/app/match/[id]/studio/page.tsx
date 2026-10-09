"use client";
import { use } from "react";
import StudioLoader from "@/components/studio/Studio";

export default function StudioPage({ params, searchParams }: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ part?: string; t?: string; moment?: string }>;
}) {
  const { id } = use(params);
  const q = use(searchParams);
  return <StudioLoader key={`${id}-${q.part ?? ""}`} match={id} part={q.part} startT={q.t != null ? Number(q.t) : undefined} momentId={q.moment} />;
}
