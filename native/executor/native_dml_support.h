// Small development-only D3D12/DirectML harness support. No ORT or Python runtime.
#pragma once
#define DML_DECLARE_INTERFACE(iid)
#define DML_TARGET_VERSION_USE_LATEST
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <DirectML.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static const GUID IID_DML_Device = {0x6dbd6437, 0x96fd, 0x423f, {0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f}};
static const GUID IID_DML_Operator = {0x26caae7a, 0x3081, 0x4633, {0x95, 0x81, 0x22, 0x6f, 0xbe, 0x57, 0x69, 0x5d}};
static const GUID IID_DML_CompiledOperator = {0x6b15e56a, 0xbf5c, 0x4902, {0x92, 0xd8, 0xda, 0x3a, 0x65, 0x0a, 0xfe, 0xa4}};
static const GUID IID_DML_CommandRecorder = {0xe6857a76, 0x2e3e, 0x4fdd, {0xbf, 0xf4, 0x5d, 0x2b, 0xa1, 0x0f, 0xb4, 0x53}};
static const GUID IID_DML_BindingTable = {0x29c687dc, 0xde74, 0x4e3b, {0xab, 0x00, 0x11, 0x68, 0xf2, 0xfc, 0x3c, 0xfc}};
static const GUID IID_D3D12_Device = {0x189819f1, 0x1db6, 0x4b57, {0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7}};
static const GUID IID_D3D12_Queue = {0x0ec870a6, 0x5d7e, 0x4c22, {0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed}};
static const GUID IID_D3D12_Allocator = {0x6102dee4, 0xaf59, 0x4b09, {0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24}};
static const GUID IID_D3D12_CommandList = {0x5b160d0f, 0xac1b, 0x4185, {0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55}};
static const GUID IID_D3D12_Fence = {0x0a753dcf, 0xc4d8, 0x4b91, {0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76}};
static const GUID IID_D3D12_QueryHeap = {0x0d9658ae, 0xed45, 0x469e, {0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4}};
static const GUID IID_D3D12_DescriptorHeap = {0x8efb471d, 0x616c, 0x4f49, {0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51}};
static const GUID IID_D3D12_Resource = {0x696442be, 0xa72e, 0x4059, {0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad}};
static const GUID IID_DXGI_Factory2 = {0x50c83a1c, 0xe072, 0x4c48, {0x87, 0xb0, 0x36, 0x30, 0xfa, 0x36, 0xa6, 0xd0}};

template<class T> static void com_release(T*& p){if(p){p->Release();p=nullptr;}}
struct GpuBuffer{ID3D12Resource* res=nullptr;uint64_t size=0;bool owns=true;~GpuBuffer(){if(owns)com_release(res);}};
struct Harness{
    HMODULE dxgi_library=nullptr,d3d12_library=nullptr,dml_library=nullptr;
    ID3D12Device* device=nullptr;IDMLDevice* dml=nullptr;
    ID3D12CommandQueue* queue=nullptr;ID3D12CommandAllocator* allocator=nullptr;
    ID3D12GraphicsCommandList* cmdlist=nullptr;ID3D12Fence* fence=nullptr;
    HANDLE event=nullptr;uint64_t fence_val=0;ID3D12QueryHeap* qheap=nullptr;
    uint64_t ts_freq=0;IDMLCommandRecorder* recorder=nullptr;uint64_t buffer_bytes_total=0,shared_budget_bytes=0;
    unsigned operator_index=0;
};
template<class T> struct ScopedCom {T* value=nullptr;~ScopedCom(){com_release(value);}};
static HMODULE load_library_utf8(const std::string& path){
    int count=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path.c_str(),-1,nullptr,0);
    if(count<=0)throw std::runtime_error("invalid UTF8 runtime path");std::vector<wchar_t> wide(size_t(count),0);
    if(!MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path.c_str(),-1,wide.data(),count))throw std::runtime_error("runtime path conversion failed");
    return LoadLibraryExW(wide.data(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}
static uint64_t align_up(uint64_t v,uint64_t a){return(v+a-1)/a*a;}
static void create_buffer(ID3D12Device* device,uint64_t bytes,D3D12_RESOURCE_STATES state,D3D12_HEAP_TYPE heap,GpuBuffer* out){
    uint64_t size=align_up(bytes?bytes:256,256);
    D3D12_HEAP_PROPERTIES hp{};hp.Type=heap;D3D12_RESOURCE_DESC d{};
    d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=size;d.Height=1;d.DepthOrArraySize=1;d.MipLevels=1;
    d.Format=DXGI_FORMAT_UNKNOWN;d.SampleDesc={1,0};d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags=heap==D3D12_HEAP_TYPE_DEFAULT?D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS:D3D12_RESOURCE_FLAG_NONE;
    HRESULT result=device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&d,state,nullptr,IID_D3D12_Resource,reinterpret_cast<void**>(&out->res));
    if(FAILED(result)||!out->res)throw std::runtime_error("CreateCommittedResource failed");out->size=size;
}
static std::vector<uint8_t> read_file(const std::string& path){
    std::ifstream stream(path,std::ios::binary|std::ios::ate);if(!stream)throw std::runtime_error("cannot read "+path);
    auto length=stream.tellg();if(length<0||uint64_t(length)>(1ull<<30))throw std::runtime_error("file outside bounded1GiB prototype: "+path);
    std::vector<uint8_t> bytes(static_cast<size_t>(length));stream.seekg(0);
    if(!bytes.empty()&&!stream.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(bytes.size())))throw std::runtime_error("short read: "+path);return bytes;
}
static void write_file(const std::string& path,const std::vector<uint8_t>& bytes){std::ofstream stream(path,std::ios::binary);if(!stream||(!bytes.empty()&&!stream.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()))))throw std::runtime_error("cannot write "+path);}
static float half_to_float(uint16_t h){
    uint32_t sign=uint32_t(h&0x8000)<<16,exp=(h>>10)&31,mant=h&1023,bits;
    if(exp==31)bits=sign|0x7f800000|(mant<<13);
    else if(exp==0){float f=std::ldexp(float(mant)/1024.0f,-14);std::memcpy(&bits,&f,4);bits|=sign;}
    else bits=sign|((exp+112)<<23)|(mant<<13);
    float f;std::memcpy(&f,&bits,4);return f;
}
static std::string narrow(const wchar_t* input){
    int length=WideCharToMultiByte(CP_UTF8,0,input,-1,nullptr,0,nullptr,nullptr);
    if(length<=0)throw std::runtime_error("adapter name conversion failed");std::vector<char> buffer(size_t(length),0);
    if(!WideCharToMultiByte(CP_UTF8,0,input,-1,buffer.data(),length,nullptr,nullptr))throw std::runtime_error("adapter name conversion failed");return std::string(buffer.data(),size_t(length-1));
}
static std::string g_out_json="native_loop_result.json";
static void json_escape(const std::string& value,std::string* out){for(char c:value){switch(c){case'"':*out+="\\\"";break;case'\\':*out+="\\\\";break;case'\n':*out+="\\n";break;case'\r':*out+="\\r";break;case'\t':*out+="\\t";break;default:*out+=c;}}}
static void emit_json(const std::string& value){std::ofstream stream(g_out_json);if(stream)stream<<value<<'\n';std::printf("NATIVE_LOOP_JSON_BEGIN\n%s\nNATIVE_LOOP_JSON_END\n",value.c_str());}
[[noreturn]] static void fail(const std::string& stage,const std::string& message,int code){std::string value="{\"status\":\"error\",\"stage\":\"";json_escape(stage,&value);value+="\",\"error\":\"";json_escape(message,&value);value+="\"}";emit_json(value);std::fprintf(stderr,"ERROR [%s] %s\n",stage.c_str(),message.c_str());std::exit(code);}
