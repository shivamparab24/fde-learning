# Setup budget: 2000 total tokens. We reserve 500 for the response.
# Remaining input budget = 1500 tokens.

from .context_assembler import ContextAssembler


assembler = ContextAssembler()
assembler = ContextAssembler(max_total_tokens=1000, reserve_output_tokens=100)
assembler = ContextAssembler(max_total_tokens=500, reserve_output_tokens=50)

# 1. System & User elements
system = "You are a senior support engineer. Always answer using retrieved docs."
user_query = "How do I fix the database replication lag?"

# 2. History 
history = [
    {"role": "user", "content": "Hi, I am having issues with my cluster."}, # Turn 1 (Oldest)
    {"role": "assistant", "content": "Hello! I can help you look into that. What kind of symptoms are you seeing?"}, # Turn 2
    {"role": "user", "content": "Our database cluster is running slow and some read queries are returning stale data."}, # Turn 3
    {"role": "assistant", "content": "Stale data on reads usually points to replica synchronization lag. Can you check the replication lag status?"}, # Turn 4
    {"role": "user", "content": "Yes, I checked the metrics. The replication lag is currently sitting at 45 seconds and climbing."}, # Turn 5
    {"role": "assistant", "content": "A 45-second lag is definitely high enough to cause stale reads. Let's dig into what is stalling the replica. Is the primary under heavy write load, or are you seeing high CPU on the replica?"}, # Turn 6
    {"role": "user", "content": "The primary CPU is around 40%, but the read replica CPU is pegged at 98%. The network throughput looks normal though."}, # Turn 7
    {"role": "assistant", "content": "Ah, if the replica CPU is maxed out while network throughput is steady, the replica is likely struggling to keep up with applying the Write-Ahead Logs (WAL) due to intensive read queries competing for resources, or explicit lock conflicts."}, # Turn 8
    {"role": "user", "content": "That makes sense. We did recently spin up a new analytics dashboard that runs heavy reporting queries directly against that read replica."}, # Turn 9
    {"role": "assistant", "content": "That is highly likely the culprit. Heavy long-running queries on a replica can trigger 'conflict with recovery' errors or cause the startup process to stall while waiting for locks. Let's look into your conflict settings and query optimization."} # Turn 10 (Newest)
]

# 3. Retrieved Documents
docs = [
    {
        "text": "Replication lag fix: Scale read replicas vertically to handle higher read loads or check network throughput between the primary and standby instances.", 
        "relevance": 0.98
    },
    {
        "text": "How to fix replication lag: Optimize slow-running queries on the primary node. Long-running transactions or write-heavy bulk updates hold locks and prevent the replica from applying changes quickly.", 
        "relevance": 0.95
    },
    {
        "text": "Network isolation: Ensure security groups allow traffic on port 5432 between instances. High packet loss on the VPC network will degrade replica synchronization rates significantly.", 
        "relevance": 0.88
    },
    {
        "text": "Handling Replication Conflicts: When a long-running read query on a standby instance conflicts with the application of a WAL record from the primary (e.g., dropping a table or vacuuming a row), the standby will pause WAL application until max_standby_streaming_delay is reached, causing replication lag to climb.",
        "relevance": 0.96
    },
    {
        "text": "Read Replica CPU Saturation: High CPU utilization on read replicas is frequently driven by missing indexes on fields utilized by complex read operations, forcing costly sequential table scans that starve the recovery process of CPU cycles.",
        "relevance": 0.92
    },
    {
        "text": "Database performance optimization generally involves indexing frequently queried columns, vacuuming dead tuples, and sizing the shared buffers correctly.", 
        "relevance": 0.60
    },
    {
        "text": "Configuring Streaming Delays: In postgresql.conf, adjusting max_standby_streaming_delay allows you to balance query completion vs. replication freshness. Setting it to a higher value allows reporting queries to finish but permits replication lag to rise temporarily during heavy write periods.",
        "relevance": 0.89
    },
    {
        "text": "Log Analysis for Recovery Conflicts: Check the PostgreSQL error logs on the standby server for messages containing 'ERROR: canceling statement due to conflict with recovery'. This confirms that incoming WAL modifications forced the system to terminate a user query to prevent indefinite replication lag.",
        "relevance": 0.85
    },
    {
        "text": "Emergency recovery: If a replica falls too far behind the primary checkpoint (past the WAL segment retention size), you must rebuild the replica from a physical backup using pg_basebackup.", 
        "relevance": 0.45
    },
    {
        "text": "To setup database replication from scratch on a new instance: 1. Edit postgresql.conf to enable hot_standby. 2. Run replica-init.sh with primary credentials. 3. Start the service.", 
        "relevance": 0.30
    },
    {
        "text": "Connection Pooling on standbys: Using an external connection pooler like PgBouncer on read replicas helps manage high connection spikes from dashboard tools, preventing backend process forks from consuming all available system memory and CPU.",
        "relevance": 0.55
    },
    {
        "text": "Hot Standby Feedback configuration: Enabling hot_standby_feedback = on on the standby informs the primary node about queries currently running on the replica. This prevents the primary from vacuuming dead rows that the replica still needs, drastically reducing recovery conflicts at the cost of slight bloat on the primary.",
        "relevance": 0.91
    },
    {
        "text": "Monitoring Replication Lag with SQL: Run 'SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn) AS sent_lag_bytes FROM pg_stat_replication;' on the primary to determine the exact byte volume differences between the primary WAL generation and what has been sent to the replicas.",
        "relevance": 0.78
    },
    {
        "text": "Disk I/O Bottlenecks on Standby: Even if network throughput is fine, slow underlying EBS volumes or storage disks on the replica can throttle WAL write speeds. Look at write IOPs and disk utilization metrics on the standby instance.",
        "relevance": 0.82
    },
    {
        "text": "Vacuum and Bloat Interactions: High autovacuum activity on the primary node generates large amounts of cleanup WAL records. When these cleanup records stream to the replica, they frequently conflict with long-running select statements, triggering the query termination protocol.",
        "relevance": 0.87
    },
    {
        "text": "WAL Configuration parameters: Increasing max_wal_size and min_wal_size on the primary node prevents frequent automatic checkpoints, smoothing out spikes in disk I/O that could otherwise starve the replication stream stream transmission rates.",
        "relevance": 0.40
    },
    {
        "text": "Diagnosing Replica Lock Contention: Query the pg_locks and pg_stat_activity catalog views on the standby server to find out if the recovery process (startup process) is being explicitly blocked by an exclusive lock held by an application backend connection.",
        "relevance": 0.90
    },
    {
        "text": "Understanding Replication Slots: Physical replication slots ensure that the primary node keeps WAL files until all replicas have consumed them. While this prevents replica failure from missing data, a stalled replica can cause disk space to fill rapidly on the primary.",
        "relevance": 0.65
    },
    {
        "text": "Tuning work_mem for Analytics: Increasing work_mem on the standby instance allows complex dashboard queries to perform sorting and hash operations entirely in RAM rather than writing temporary files to disk, freeing up disk I/O for the replication stream.",
        "relevance": 0.73
    },
    {
        "text": "Cascading Replication architecture: For massive dashboard read loads, consider setting up cascading replication where a primary streams to a single upstream replica, which then distributes the data stream out to multiple downstream read-only nodes.",
        "relevance": 0.50
    }
]

# Assemble
result = assembler.assemble(
    system_prompt=system,
    user_query=user_query,
    history=history,
    retrieved_docs=docs
)

# Output structural telemetry
import json
print(json.dumps(result["metrics"], indent=2))