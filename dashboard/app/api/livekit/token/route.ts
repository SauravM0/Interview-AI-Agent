import { AccessToken } from "livekit-server-sdk";
import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const roomName = searchParams.get("roomName");
  const participantName = searchParams.get("participantName");

  if (!roomName || !participantName) {
    return NextResponse.json({ error: "Missing roomName or participantName" }, { status: 400 });
  }

  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;
  if (!apiKey || !apiSecret) {
    return NextResponse.json({ error: "LiveKit server not configured" }, { status: 500 });
  }

  // Unique identity prevents participant collisions in the same room
  const identity = `${participantName.toLowerCase().replace(/\s+/g, "-")}-${Date.now()}`;

  const at = new AccessToken(apiKey, apiSecret, { identity, name: participantName, ttl: "1h" });
  at.addGrant({ roomJoin: true, room: roomName, canPublish: true, canSubscribe: true });

  return NextResponse.json({ token: await at.toJwt(), identity });
}
