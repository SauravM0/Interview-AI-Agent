import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const roomName = searchParams.get("roomName");
  const participantName = searchParams.get("participantName");

  if (!roomName || !participantName) {
    return NextResponse.json(
      { error: "Missing roomName or participantName" },
      { status: 400 }
    );
  }

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const tokenUrl =
    `${apiUrl}/api/interviews/generate-token` +
    `?room_name=${encodeURIComponent(roomName)}` +
    `&identity=${encodeURIComponent(participantName)}` +
    `&name=${encodeURIComponent(participantName)}`;

  const res = await fetch(tokenUrl, { cache: "no-store" });
  const body = await res.text();

  return new NextResponse(body, {
    status: res.status,
    headers: {
      "Content-Type": res.headers.get("Content-Type") || "application/json",
    },
  });
}
