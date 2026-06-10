import { createUser } from "../services/userService";

export async function POST(request: Request) {
  const body = await request.json();
  const user = await createUser(body);
  return Response.json({ id: user.id, email: user.email });
}

