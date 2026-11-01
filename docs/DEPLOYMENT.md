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

The [live demo](https://medical-speech-lab.65.0.103.235.sslip.io/) uses the existing `saga-agent-demo` EC2 instance (`i-038a14281fb8c141c`) and its Caddy proxy. [`shared-compose.yaml`](../deploy/shared-compose.yaml) starts only the speech API on the existing `saga-agent_default` Docker network. It publishes no host ports. The [Caddy site block](../deploy/shared-Caddyfile.block) adds `medical-speech-lab.65.0.103.235.sslip.io` to the host proxy configuration. The SagaAgent and ReorderIQ site blocks are retained.

On the host, clone this repository to `/opt/medical-speech-lab` and pin the intended commit. Check `docker compose -f deploy/shared-compose.yaml config`, then build and start:

```bash
cd /opt/medical-speech-lab
docker compose -f deploy/shared-compose.yaml up -d --build
docker compose -f deploy/shared-compose.yaml ps
```

The host Caddyfile is `/opt/saga/deploy/aws/Caddyfile`, with a pre-change backup at `/opt/saga/deploy/aws/Caddyfile.before-medical-speech-lab-20261101`. Validate the complete host file with a temporary Caddy container that has `SAGA_DOMAIN` set. **Recreate only the proxy** from `/opt/saga/deploy/aws` using `docker compose up -d --no-deps --force-recreate proxy` after changing the host file. The existing proxy's single-file bind mount can point at an older inode after that file is replaced, so `docker exec ... caddy reload` alone may load stale routes. Recreating the proxy remounts the current file and briefly interrupts HTTPS; it does not recreate the apps. Verify SagaAgent, ReorderIQ, and Medical Speech Lab after the change.

A 2 GB EBS-backed swapfile was added to the already provisioned root volume to protect this 2 GB instance from build-time memory pressure. The shared API is capped at 900 MB and one CPU. After a real transcription, it used about 422 MiB, with about 740 MiB host memory available. Monitor `docker stats`, `free -m`, and disk use during updates. The image was built from commit `3f85e15bcd54b2d13879653e039de0ebf09d45ec`. No separate EC2 instance or public IP was created.

To remove only this demo, remove the new site block from the host Caddyfile while preserving the other two routes, validate the complete file, recreate only the proxy as above, and run `docker compose -f deploy/shared-compose.yaml down` in this repository. This keeps SagaAgent, ReorderIQ, and the EC2 instance running. Retain the Caddyfile backup until all sites are healthy. If another deployment replaces the host Caddyfile, reapply the speech block before recreating the proxy.

## Operational scope

The live demo is intended for low traffic and public, non-sensitive recordings. It has process-level concurrency control and a per-IP rate limit, with no queue, account system, audit logs, data retention, SLA, or clinical validation. Do not submit patient information. Clinical deployment would require a separate privacy, security, quality, and regulatory review.
