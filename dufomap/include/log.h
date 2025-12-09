#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

static FILE* g_logFile = nullptr;

int g_logf(const char* format, ...)
{
	va_list args;
	va_start(args, format);
	char buffer[2048];
	vsnprintf(buffer, sizeof(buffer), format, args);
	va_end(args);
	if (g_logFile) {
		fprintf(g_logFile, "%s", buffer);
		fflush(g_logFile);
	}
	return printf("%s", buffer);
}

int g_log_init(std::string log_dir, std::string &str_out) {
	//获取系统时间  
	time_t now_time = time(NULL);
	tm t_tm = tm();

#if defined I_OS_LINUX
	localtime_r(&now_time, &t_tm);
#else
	localtime_s(&t_tm, &now_time);
#endif

	char tmp[32]{ 0 };
	snprintf(tmp, sizeof(tmp), "%04d-%02d-%02d__%02d-%02d-%02d", 1900 + t_tm.tm_year, 1 + t_tm.tm_mon, t_tm.tm_mday, t_tm.tm_hour, t_tm.tm_min, t_tm.tm_sec);

	std::string str_time(tmp);
	std::string log_file_path = log_dir + "log_" + str_time + ".txt";
	str_out = str_time;

#ifdef _MSC_VER
	if (fopen_s(&g_logFile, log_file_path.c_str(), "w") != 0) {
		g_logFile = nullptr;
		return -1;
	}
		
#else
	g_logFile = fopen(log_file_path.c_str(), "w");
#endif

	return 0;
}
