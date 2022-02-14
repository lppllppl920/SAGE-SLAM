/**
 * ****************************************************************************
 * Copyright (c) 2017, Robert Lukierski.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * ****************************************************************************
 * C++ wrapper for C file IO.
 * ****************************************************************************
 */

#ifndef VISIONCORE_IO_FILE_HPP
#define VISIONCORE_IO_FILE_HPP

#include <cstdio>
#include <cstdarg>
#include <memory>

namespace vc
{

namespace io
{
  
class File
{
public:
   File() : fd(0,::fclose) { }
   
   ~File()
   {
     
   }
   
   File(const File&) = delete;
   File(File&&) = delete;
   File& operator=(const File&) = delete;
   File& operator=(File&&) = delete;
  
   File(const char* fn, const char* mode = "r") : fd(0,::fclose)
   {
       open(fn,mode);
   }
   
   bool open(const char* fn, const char* mode = "r")
   {
       fd = std::unique_ptr<FILE,int(*)(FILE*)>(::fopen(fn,mode),::fclose);
       return isOpened();
   }
   
   void close()
   {
       fd.reset();
   }
   
   void flush()
   {
       ::fflush(fd.get());
   }
   
   bool eof()
   {
       return ::feof(fd.get()) != 0;
   }
   
   bool isOpened() const { return fd.get() != NULL; }
   
   std::size_t read(void *ptr, std::size_t size, std::size_t nmemb)
   {
       return ::fread(ptr, size, nmemb, fd.get());
   }
   
   std::size_t write(const void *ptr, std::size_t size, std::size_t nmemb)
   {
       return ::fwrite(ptr, size, nmemb, fd.get());
   }
   
   int getc()
   {
       return ::fgetc(fd.get());
   }
   
   bool putc(int c)
   {
       return ::fputc(c, fd.get()) == c;
   }
   
   char* gets(char* str, std::size_t count)
   {
       return ::fgets(str, count, fd.get());
   }
   
   bool puts(const char* str)
   {
       return ::fputs(str, fd.get()) > 0;
   }
   
   int scanf(const char* fmt, ...)
   {
       va_list ap;
       va_start(ap, fmt);
       int rc = vfscanf(fd.get(), fmt, ap);
       va_end(ap);
       return rc;
   }
   
   int printf(const char* fmt, ...)
   {
       va_list ap;
       va_start(ap, fmt);
       int rc = vfprintf(fd.get(), fmt, ap);
       va_end(ap);
     
       return rc;
   }
private:
   std::unique_ptr<FILE,int(*)(FILE*)> fd;
};

}

}

#endif // VISIONCORE_IO_FILE_HPP
