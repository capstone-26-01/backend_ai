def send_analysis_ready_email(user, repo_id):
    return {"to": user.email, "repo_id": repo_id}


def render_template(name, context):
    return f"{name}:{sorted(context)}"
