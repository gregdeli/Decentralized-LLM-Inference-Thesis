"""----- LLM Constants -----"""
GLOBAL_MAX_SEQ_LEN = 8192 #2048 #4096

"""----- Server and Client Constants -----"""
RESERVED_MEM_MB = 500 # Memory reserved for system overhead

REALLOC_TAKEOVER_MULT_THRESHOLD = 4.0

OPPRTUNISTIC_TAKEOVER_MULT_THRESHOLD = 1.5

PROFILING_DURATION = 4 # seconds

"""----- GRPC Constants -----"""
GRPC_MAX_MSG_SIZE = 100 * 1024 * 1024  # 100 MB

GPRC_MAX_WORKERS = 2