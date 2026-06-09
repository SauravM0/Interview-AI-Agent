import { InterviewRoom } from './InterviewRoom';

interface PageProps {
    params: Promise<{ roomId: string }>;
}

export default async function Page({ params }: PageProps) {
    const { roomId } = await params;
    const livekitUrl = process.env.LIVEKIT_URL || "ws://localhost:7880";

    return <InterviewRoom roomId={roomId} url={livekitUrl} />;
}
