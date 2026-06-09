import { InterviewRoom } from './InterviewRoom';

interface PageProps {
    params: Promise<{ roomId: string }>;
}

export default async function Page({ params }: PageProps) {
    const { roomId } = await params;

    return <InterviewRoom roomId={roomId} />;
}
