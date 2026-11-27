# Hugging Face Space (Docker SDK). The Gradio SDK image pins gradio[mcp] to mcp<2,
# which conflicts with this app's MCP server (mcp>=2), so the Space runs its own image.
FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 PORT=7860 \
    HF_HUB_DISABLE_XET=1 GRADIO_ANALYTICS_ENABLED=False
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
