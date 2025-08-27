"""
Logging Configuration Utilities

This module provides centralized logging configuration for the fraud detection system
with proper formatting, file rotation, and log level management.

 Version: 1.0.0
"""

import os
import time
import logging
import logging.handlers
from typing import Dict, Any, Optional
from datetime import datetime

from ..config import LOGGING_CONFIG


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Set up application logging with proper configuration.
    
    Args:
        config (Optional[Dict[str, Any]]): Custom logging configuration
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Configure root logger with appropriate level
        - Set up file handlers with rotation
        - Configure console handler for development
        - Apply consistent formatting across handlers
        - Create log directory if needed
    """
    if config is None:
        config = LOGGING_CONFIG
    
    try:
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root logger level
        log_level = getattr(logging, config.get('level', 'INFO').upper())
        root_logger.setLevel(log_level)
        
        # Create log directory if needed
        log_file = config.get('log_file')
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Create file handler with rotation
            file_handler = create_file_handler(log_file, config.get('level', 'INFO'))
            root_logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = create_console_handler('DEBUG' if config.get('debug', False) else 'INFO')
        root_logger.addHandler(console_handler)
        
        logger = logging.getLogger(__name__)
        logger.info("Logging system initialized successfully")
        
    except Exception as e:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        logger.error(f"Error setting up logging: {str(e)}, using basic configuration")


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name (str): Logger name (usually __name__)
        level (Optional[str]): Override log level
        
    Returns:
        logging.Logger: Configured logger instance
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Create logger with specified name
        - Apply consistent configuration
        - Set appropriate log level
        - Ensure proper handler attachment
    """
    logger = logging.getLogger(name)
    
    # Set custom level if provided
    if level:
        try:
            log_level = getattr(logging, level.upper())
            logger.setLevel(log_level)
        except AttributeError:
            logger.setLevel(logging.INFO)
            logger.warning(f"Invalid log level '{level}', using INFO")
    
    # Ensure logger inherits from root logger if no handlers
    if not logger.handlers:
        logger.parent = logging.getLogger()
    
    return logger


def create_file_handler(log_file: str, level: str = 'INFO') -> logging.Handler:
    """
    Create a file handler with rotation.
    
    Args:
        log_file (str): Path to log file
        level (str): Log level for this handler
        
    Returns:
        logging.Handler: Configured file handler
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Create RotatingFileHandler with size limits
        - Set appropriate log level
        - Apply consistent formatting
        - Handle file creation and permissions
    """
    try:
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOGGING_CONFIG.get('max_bytes', 10485760),  # 10MB default
            backupCount=LOGGING_CONFIG.get('backup_count', 5),
            encoding='utf-8'
        )
        
        # Set log level
        log_level = getattr(logging, level.upper(), logging.INFO)
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            fmt=LOGGING_CONFIG.get('format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        return file_handler
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error creating file handler: {str(e)}")
        return logging.NullHandler()


def create_console_handler(level: str = 'DEBUG') -> logging.Handler:
    """
    Create a console handler for development.
    
    Args:
        level (str): Log level for console output
        
    Returns:
        logging.Handler: Configured console handler
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Create StreamHandler for console output
        - Apply development-friendly formatting
        - Set appropriate colors if supported
        - Configure for different environments
    """
    try:
        # Create console handler
        console_handler = logging.StreamHandler()
        
        # Set log level
        log_level = getattr(logging, level.upper(), logging.DEBUG)
        console_handler.setLevel(log_level)
        
        # Create development-friendly formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        return console_handler
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error creating console handler: {str(e)}")
        return logging.NullHandler()


def log_function_call(func_name: str, args: Dict[str, Any] = None, 
                     logger: logging.Logger = None) -> None:
    """
    Log function calls for debugging.
    
    Args:
        func_name (str): Name of function being called
        args (Dict[str, Any], optional): Function arguments to log
        logger (logging.Logger, optional): Logger instance to use
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Log function entry with arguments
        - Handle sensitive data masking
        - Format arguments for readability
        - Use appropriate log level
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        if args:
            # Mask sensitive data
            masked_args = {}
            sensitive_keys = ['password', 'token', 'key', 'secret', 'auth']
            
            for key, value in args.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    masked_args[key] = '***MASKED***'
                else:
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 100:
                        masked_args[key] = value[:97] + '...'
                    else:
                        masked_args[key] = value
            
            logger.debug(f"Calling {func_name} with args: {masked_args}")
        else:
            logger.debug(f"Calling {func_name}()")
            
    except Exception as e:
        logger.error(f"Error logging function call for {func_name}: {str(e)}")


def log_performance_metrics(operation: str, duration: float, 
                          additional_metrics: Dict[str, Any] = None,
                          logger: logging.Logger = None) -> None:
    """
    Log performance metrics for operations.
    
    Args:
        operation (str): Name of operation
        duration (float): Operation duration in seconds
        additional_metrics (Dict[str, Any], optional): Additional metrics
        logger (logging.Logger, optional): Logger instance to use
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Log operation performance with duration
        - Include additional metrics if provided
        - Format metrics for analysis
        - Use structured logging format
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Create structured performance log entry
        metrics = {
            'operation': operation,
            'duration_seconds': round(duration, 3),
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Log with structured format for analysis
        logger.info(f"PERFORMANCE: {operation} completed in {duration:.3f}s | {metrics}")
        
        # Log warning for slow operations
        if duration > 5.0:  # Configurable threshold
            logger.warning(f"SLOW OPERATION: {operation} took {duration:.3f}s")
            
    except Exception as e:
        logger.error(f"Error logging performance metrics for {operation}: {str(e)}")


def log_error_with_context(error: Exception, context: Dict[str, Any] = None,
                          logger: logging.Logger = None) -> None:
    """
    Log errors with contextual information.
    
    Args:
        error (Exception): Exception to log
        context (Dict[str, Any], optional): Contextual information
        logger (logging.Logger, optional): Logger instance to use
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Log exception with full traceback
        - Include contextual information
        - Mask sensitive data in context
        - Use appropriate error formatting
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        import traceback
        
        # Create error context
        error_context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            # Mask sensitive data
            masked_context = {}
            sensitive_keys = ['password', 'token', 'key', 'secret', 'auth', 'credential']
            
            for key, value in context.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    masked_context[key] = '***MASKED***'
                else:
                    masked_context[key] = value
            
            error_context['context'] = masked_context
        
        # Log error with context
        logger.error(f"ERROR: {type(error).__name__}: {str(error)}")
        
        if context:
            logger.error(f"ERROR CONTEXT: {error_context['context']}")
        
        # Log full traceback at debug level
        logger.debug(f"ERROR TRACEBACK:\n{traceback.format_exc()}")
        
    except Exception as e:
        logger.error(f"Error in error logging: {str(e)} (Original error: {str(error)})")
        logger.debug(traceback.format_exc())


def create_audit_log(action: str, user_id: str = None, 
                    data: Dict[str, Any] = None) -> None:
    """
    Create audit log entries for important actions.
    
    Args:
        action (str): Action being performed
        user_id (str, optional): User performing action
        data (Dict[str, Any], optional): Additional data to log
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Create separate audit log file
        - Log with timestamp and user information
        - Include action details and data
        - Ensure audit log integrity
    """
    try:
        # Get or create audit logger
        audit_logger = logging.getLogger('audit')
        
        # Set up audit file handler if not exists
        if not audit_logger.handlers:
            audit_file = LOGGING_CONFIG.get('audit_log_file', 'logs/audit.log')
            os.makedirs(os.path.dirname(audit_file), exist_ok=True)
            
            audit_handler = logging.handlers.RotatingFileHandler(
                audit_file,
                maxBytes=5242880,  # 5MB
                backupCount=10,
                encoding='utf-8'
            )
            audit_handler.setLevel(logging.INFO)
            
            # Use structured format for audit logs
            formatter = logging.Formatter(
                '%(asctime)s - AUDIT - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            audit_handler.setFormatter(formatter)
            audit_logger.addHandler(audit_handler)
            audit_logger.setLevel(logging.INFO)
            audit_logger.propagate = False  # Don't propagate to root logger
        
        # Create audit entry
        audit_entry = {
            'action': action,
            'user_id': user_id or 'system',
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        # Log structured audit entry
        audit_logger.info(f"ACTION: {action} | USER: {user_id or 'system'} | DATA: {data or {}}")
        
    except Exception as e:
        # Fallback to regular logger if audit logging fails
        logger = logging.getLogger(__name__)
        logger.error(f"Audit logging failed for action '{action}': {str(e)}")


def setup_request_logging() -> None:
    """
    Set up HTTP request logging for web application.
    
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Configure request/response logging
        - Log request details and timing
        - Handle sensitive data masking
        - Integrate with web framework logging
    """
    try:
        # Create request logger
        request_logger = logging.getLogger('requests')
        
        if not request_logger.handlers:
            request_file = LOGGING_CONFIG.get('request_log_file', 'logs/requests.log')
            os.makedirs(os.path.dirname(request_file), exist_ok=True)
            
            request_handler = logging.handlers.RotatingFileHandler(
                request_file,
                maxBytes=10485760,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            request_handler.setLevel(logging.INFO)
            
            # Format for request logs
            formatter = logging.Formatter(
                '%(asctime)s - REQUEST - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            request_handler.setFormatter(formatter)
            request_logger.addHandler(request_handler)
            request_logger.setLevel(logging.INFO)
            request_logger.propagate = False
        
        logger = logging.getLogger(__name__)
        logger.info("Request logging configured successfully")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error setting up request logging: {str(e)}")


def cleanup_old_logs(days_to_keep: int = 30) -> int:
    """
    Clean up old log files.
    
    Args:
        days_to_keep (int): Number of days to retain logs
        
    Returns:
        int: Number of files cleaned up
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Find old log files based on modification date
        - Remove files older than specified days
        - Handle cleanup errors gracefully
        - Log cleanup operations
    """
    logger = logging.getLogger(__name__)
    files_cleaned = 0
    
    try:
        log_file = LOGGING_CONFIG.get('log_file')
        if not log_file:
            logger.info("No log file configured for cleanup")
            return 0
        
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            logger.info(f"Log directory does not exist: {log_dir}")
            return 0
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        # Find and remove old log files
        for filename in os.listdir(log_dir):
            filepath = os.path.join(log_dir, filename)
            
            # Only process log files
            if filename.endswith('.log') or filename.endswith('.log.1') or '.log.' in filename:
                try:
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        files_cleaned += 1
                        logger.debug(f"Removed old log file: {filename}")
                except OSError as e:
                    logger.warning(f"Could not remove log file {filename}: {str(e)}")
        
        if files_cleaned > 0:
            logger.info(f"Log cleanup completed: removed {files_cleaned} old log files")
        
        return files_cleaned
        
    except Exception as e:
        logger.error(f"Error during log cleanup: {str(e)}")
        return 0


def configure_production_logging() -> None:
    """
    Configure logging for production environment.
    
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Set production-appropriate log levels
        - Configure centralized logging if needed
        - Set up log aggregation
        - Ensure performance optimization
    """
    try:
        # Production logging configuration
        prod_config = {
            'level': 'INFO',  # Less verbose than DEBUG
            'log_file': 'logs/fraud_detector_prod.log',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'max_bytes': 20971520,  # 20MB for production
            'backup_count': 10,
            'debug': False
        }
        
        # Apply production configuration
        setup_logging(prod_config)
        
        # Set up additional production loggers
        setup_request_logging()
        
        # Configure third-party library logging
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('selenium').setLevel(logging.WARNING)
        
        logger = logging.getLogger(__name__)
        logger.info("Production logging configuration applied")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error configuring production logging: {str(e)}")


def get_logging_stats() -> Dict[str, Any]:
    """
    Get statistics about logging activity.
    
    Returns:
        Dict[str, Any]: Logging statistics
        
    Implementation by Orchestration Engineer - Infrastructure & Deployment:
        - Count log entries by level
        - Calculate log file sizes
        - Track logging performance
        - Return comprehensive statistics
    """
    try:
        stats = {
            'handlers': [],
            'log_files': {},
            'loggers': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Get handler information
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler_info = {
                'type': type(handler).__name__,
                'level': handler.level,
                'level_name': logging.getLevelName(handler.level)
            }
            
            # Add file-specific info for file handlers
            if hasattr(handler, 'baseFilename'):
                handler_info['filename'] = handler.baseFilename
                try:
                    file_size = os.path.getsize(handler.baseFilename)
                    handler_info['file_size_bytes'] = file_size
                    handler_info['file_size_mb'] = round(file_size / 1024 / 1024, 2)
                except (OSError, AttributeError):
                    handler_info['file_size_bytes'] = 0
            
            stats['handlers'].append(handler_info)
        
        # Get log file information
        log_file = LOGGING_CONFIG.get('log_file')
        if log_file and os.path.exists(log_file):
            try:
                stats['log_files']['main'] = {
                    'path': log_file,
                    'size_bytes': os.path.getsize(log_file),
                    'size_mb': round(os.path.getsize(log_file) / 1024 / 1024, 2),
                    'modified': datetime.fromtimestamp(os.path.getmtime(log_file)).isoformat()
                }
            except OSError:
                pass
        
        # Get active loggers
        stats['loggers'] = list(logging.Logger.manager.loggerDict.keys())
        stats['logger_count'] = len(stats['loggers'])
        
        return stats
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting logging stats: {str(e)}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }