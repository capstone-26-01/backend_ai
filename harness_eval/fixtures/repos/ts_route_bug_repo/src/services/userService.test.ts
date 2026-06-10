import { createUser } from "./userService";

test("creates a user", async () => {
  const user = await createUser({ email: "ada@example.com" });
  expect(user.id).toBe("user_123");
});

