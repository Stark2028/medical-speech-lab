# Deployment

The public demo runs one CPU Whisper tiny baseline model behind Caddy HTTPS. The model is baked into a Docker image at a pinned Hub revision; uploads and transcripts are not written to disk. The service is a **research demo**, not a clinical product. Source dataset files and fine-tuned weights are not deployed.

## Local API

Install CPU PyTorch and API dependencies, then run:

```powershell
python -m pip install torch==2.12.0
python -m pip install -r requirements-api.txt
uvicorn medspeech.service:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. `/health/ready` is healthy only after the model loads. The same-origin API accepts a raw WAV request body at `POST /api/v1/transcribe` with `Content-Type: audio/wav`; it returns JSON. Uploaded files are limited to 8 MB and 30 seconds; the model handles one request at a time. Concurrent requests and requests beyond 12 per IP per minute return HTTP 429. The in-process limiter assumes one API worker and Caddy as the sole public entry point.

## Docker Compose

Set `SITE_HOST` in ignored `deploy/.env` to a hostname that points to the server. The host must allow ports 80 and 443. Then:

```bash
cd deploy
docker compose up -d --build
docker compose ps
```

Caddy provisions and renews TLS, restricts the request body to 8 MB, and proxies only to the internal API service. The API container is read-only, runs as a non-root user, has one model worker, and has a readiness health check. If using `sslip.io`, the host can be `medical-speech-lab.<public-ip>.sslip.io`.

## Dedicated AWS EC2 stack

`deploy/aws.yaml` provisions a dedicated Amazon Linux 2023 instance, encrypted 30 GB root volume, Elastic IP, a security group exposing only ports 80/443, and an SSM instance role. Cloud-init installs Docker/Compose, clones the public repository, builds the image, and starts the stack. There is no SSH port or key. **An EC2 instance, EBS volume, and public IPv4 address incur ongoing AWS charges until the CloudFormation stack is deleted.**

The repository must be public before stack creation. Select a VPC and public subnet in `ap-south-1`, then deploy:

```powershell
aws cloudformation deploy --region ap-south-1 --stack-name medical-speech-lab-demo --template-file deploy/aws.yaml --capabilities CAPABILITY_IAM --parameter-overrides VpcId=vpc-xxxxxxxx SubnetId=subnet-xxxxxxxx InstanceType=t3.medium
aws cloudformation describe-stacks --region ap-south-1 --stack-name medical-speech-lab-demo --query "Stacks[0].Outputs" --output table
```

The first boot builds the image and downloads the pinned baseline model, so readiness may take several minutes. Inspect boot logs with AWS Systems Manager Session Manager (`sudo tail -f /var/log/cloud-init-output.log`) and service logs with `cd /opt/medical-speech-lab/deploy && sudo docker compose logs -f`.

To update after a tested commit:

```bash
cd /opt/medical-speech-lab
sudo git pull --ff-only
cd deploy
sudo docker compose up -d --build
sudo docker image prune -f
```

To stop charges and remove the demo:

```powershell
aws cloudformation delete-stack --region ap-south-1 --stack-name medical-speech-lab-demo
```

## Shared SagaAgent EC2 host

The selected live deployment uses the existing `saga-agent-demo` EC2 instance (`i-038a14281fb8c141c`) and its Caddy proxy. [`shared-compose.yaml`](../deploy/shared-compose.yaml) starts only the speech API on the existing `saga-agent_default` Docker network. It publishes no host ports. The [Caddy site block](../deploy/shared-Caddyfile.block) adds `medical-speech-lab.65.0.103.235.sslip.io` to the existing proxy configuration. The SagaAgent site block is retained.

On the host, clone this repository to `/opt/medical-speech-lab` and pin the intended commit. Check `docker compose -f deploy/shared-compose.yaml config`, then build and start:

```bash
cd /opt/medical-speech-lab
docker compose -f deploy/shared-compose.yaml up -d --build
docker compose -f deploy/shared-compose.yaml ps
```

Back up `/opt/saga/deploy/aws/Caddyfile` before appending the new site block. Validate with `docker exec saga-agent-proxy-1 caddy validate --config /etc/caddy/Caddyfile`, then reload with `docker exec saga-agent-proxy-1 caddy reload --config /etc/caddy/Caddyfile`. Verify both the new hostname and the existing SagaAgent hostname after reload. A 2 GB EBS-backed swapfile can protect the 2 GB instance from build-time memory pressure; monitor `docker stats` and `free -m`. The shared API is capped at 900 MB and one CPU. Building the image may temporarily consume CPU and disk.

To remove only this demo, remove the new site block from Caddyfile, validate and reload Caddy, then run `docker compose -f deploy/shared-compose.yaml down` in this repository. This keeps SagaAgent and the EC2 instance running. Keep the Caddyfile backup until both sites are healthy.

## Operational scope

The live demo is intended for low traffic and public, non-sensitive recordings. It has process-level concurrency control and a per-IP rate limit, with no queue, account system, audit logs, data retention, SLA, or clinical validation. Do not submit patient information. Clinical deployment would require a separate privacy, security, quality, and regulatory review.
