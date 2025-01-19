# MLOps Final Project Setup Instructions

## Infrastructure Setup Documentation Requirements

For each step below, make sure to take screenshots of:
1. The configuration process
2. The final running state
3. Any verification steps

## AWS EC2 Setup

### Instance 1: Tracking Infrastructure

#### 1. EC2 Instance Creation
- Launch an EC2 instance
  - Name: `tracking-server`
  - AMI: Ubuntu Server 22.04 LTS
  - Instance type: t2.medium (recommended)
  - Key pair: Create new or use existing
  - Security Group settings:
    - Allow SSH (Port 22)
    - Allow HTTP (Port 80)
    - Allow HTTPS (Port 443)
    - Allow MLflow (Port 5000)
    - Allow PostgreSQL (Port 5432)
    - Allow MinIO (Port 9000, 9001)

**Screenshot required:** 
- EC2 launch wizard configuration
- Security group settings
- Running instance details

#### 2. PostgreSQL Setup
```bash
# Install Docker
sudo apt update
sudo apt install docker.io docker-compose -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Create docker-compose.yml for PostgreSQL
cat << EOF > docker-compose-postgres.yml
version: '3'
services:
  postgres:
    image: postgres:14
    container_name: mlflow_postgres
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow123
      POSTGRES_DB: mlflow
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: always

volumes:
  postgres_data:
EOF

# Start PostgreSQL
docker-compose -f docker-compose-postgres.yml up -d
```

**Screenshot required:**
- Docker compose file
- Running PostgreSQL container
- Connection test

#### 3. MinIO Setup
```bash
# Create docker-compose.yml for MinIO
cat << EOF > docker-compose-minio.yml
version: '3'
services:
  minio:
    image: minio/minio
    container_name: mlflow_minio
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    restart: always

volumes:
  minio_data:
EOF

# Start MinIO
docker-compose -f docker-compose-minio.yml up -d
```

**Screenshot required:**
- Docker compose file
- Running MinIO container
- MinIO console access

#### 4. MLflow Setup
```bash
# Create docker-compose.yml for MLflow
cat << EOF > docker-compose-mlflow.yml
version: '3'
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow_server
    ports:
      - "5000:5000"
    environment:
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin123
      - MLFLOW_S3_ENDPOINT_URL=http://mlflow_minio:9000
    command: >
      mlflow server 
      --host 0.0.0.0 
      --port 5000
      --backend-store-uri postgresql://mlflow:mlflow123@mlflow_postgres:5432/mlflow
      --default-artifact-root s3://mlflow/
    depends_on:
      - postgres
      - minio
    restart: always

networks:
  default:
    external:
      name: mlflow_network
EOF

# Create network and start MLflow
docker network create mlflow_network
docker network connect mlflow_network mlflow_postgres
docker network connect mlflow_network mlflow_minio
docker-compose -f docker-compose-mlflow.yml up -d
```

**Screenshot required:**
- Docker compose file
- Running MLflow container
- MLflow UI access
- Network configuration

### Instance 2 & 3: Staging and Production Servers

#### 1. EC2 Instance Creation (Repeat for both)
- Launch EC2 instances
  - Names: `staging-server` and `production-server`
  - AMI: Ubuntu Server 22.04 LTS
  - Instance type: t2.micro
  - Security Group settings:
    - Allow SSH (Port 22)
    - Allow HTTP (Port 80)
    - Allow HTTPS (Port 443)

**Screenshot required:**
- EC2 launch wizard configuration
- Security group settings
- Running instances details

#### 2. Nginx Setup (For both instances)
```bash
# Install Nginx
sudo apt update
sudo apt install nginx -y

# Configure Nginx with load balancing
sudo bash -c 'cat << EOF > /etc/nginx/sites-available/default
upstream backend {
    server localhost:8000;
    server localhost:8001;
}

server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF'

# Restart Nginx
sudo systemctl restart nginx
```

**Screenshot required:**
- Nginx configuration file
- Nginx service status
- Load balancer test results

## Required Documentation Screenshots Summary

1. Tracking Infrastructure (Instance 1):
   - EC2 Instance Creation
   - Security Groups Configuration
   - PostgreSQL Setup and Verification
   - MinIO Setup and Console Access
   - MLflow Setup and UI Access
   - Network Configuration

2. Staging Server (Instance 2):
   - EC2 Instance Creation
   - Security Groups Configuration
   - Nginx Setup and Configuration
   - Load Balancer Verification

3. Production Server (Instance 3):
   - EC2 Instance Creation
   - Security Groups Configuration
   - Nginx Setup and Configuration
   - Load Balancer Verification

## Additional Notes

1. Keep all credentials and access keys secure
2. Document any troubleshooting steps
3. Include verification steps for each service
4. Document any modifications made to default configurations
5. Include resource cleanup instructions

## Verification Steps

After setting up each component, verify:

1. PostgreSQL:
```bash
psql -h localhost -p 5432 -U mlflow -d mlflow
```

2. MinIO:
- Access console at http://<EC2-IP>:9001
- Create bucket named "mlflow"

3. MLflow:
- Access UI at http://<EC2-IP>:5000
- Verify experiment tracking works

4. Nginx:
```bash
curl -I http://localhost
```

Remember to replace `<EC2-IP>` with your actual EC2 instance IP addresses in the documentation. 