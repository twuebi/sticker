// This file is generated by rust-protobuf 2.5.0. Do not edit
// @generated

// https://github.com/Manishearth/rust-clippy/issues/702
#![allow(unknown_lints)]
#![allow(clippy)]

#![cfg_attr(rustfmt, rustfmt_skip)]

#![allow(box_pointers)]
#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(trivial_casts)]
#![allow(unsafe_code)]
#![allow(unused_imports)]
#![allow(unused_results)]

use protobuf::Message as Message_imported_for_functions;
use protobuf::ProtobufEnum as ProtobufEnum_imported_for_functions;

#[derive(PartialEq,Clone,Default)]
pub struct AllocationDescription {
    // message fields
    pub requested_bytes: i64,
    pub allocated_bytes: i64,
    pub allocator_name: ::std::string::String,
    pub allocation_id: i64,
    pub has_single_reference: bool,
    pub ptr: u64,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a AllocationDescription {
    fn default() -> &'a AllocationDescription {
        <AllocationDescription as ::protobuf::Message>::default_instance()
    }
}

impl AllocationDescription {
    pub fn new() -> AllocationDescription {
        ::std::default::Default::default()
    }

    // int64 requested_bytes = 1;


    pub fn get_requested_bytes(&self) -> i64 {
        self.requested_bytes
    }
    pub fn clear_requested_bytes(&mut self) {
        self.requested_bytes = 0;
    }

    // Param is passed by value, moved
    pub fn set_requested_bytes(&mut self, v: i64) {
        self.requested_bytes = v;
    }

    // int64 allocated_bytes = 2;


    pub fn get_allocated_bytes(&self) -> i64 {
        self.allocated_bytes
    }
    pub fn clear_allocated_bytes(&mut self) {
        self.allocated_bytes = 0;
    }

    // Param is passed by value, moved
    pub fn set_allocated_bytes(&mut self, v: i64) {
        self.allocated_bytes = v;
    }

    // string allocator_name = 3;


    pub fn get_allocator_name(&self) -> &str {
        &self.allocator_name
    }
    pub fn clear_allocator_name(&mut self) {
        self.allocator_name.clear();
    }

    // Param is passed by value, moved
    pub fn set_allocator_name(&mut self, v: ::std::string::String) {
        self.allocator_name = v;
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_allocator_name(&mut self) -> &mut ::std::string::String {
        &mut self.allocator_name
    }

    // Take field
    pub fn take_allocator_name(&mut self) -> ::std::string::String {
        ::std::mem::replace(&mut self.allocator_name, ::std::string::String::new())
    }

    // int64 allocation_id = 4;


    pub fn get_allocation_id(&self) -> i64 {
        self.allocation_id
    }
    pub fn clear_allocation_id(&mut self) {
        self.allocation_id = 0;
    }

    // Param is passed by value, moved
    pub fn set_allocation_id(&mut self, v: i64) {
        self.allocation_id = v;
    }

    // bool has_single_reference = 5;


    pub fn get_has_single_reference(&self) -> bool {
        self.has_single_reference
    }
    pub fn clear_has_single_reference(&mut self) {
        self.has_single_reference = false;
    }

    // Param is passed by value, moved
    pub fn set_has_single_reference(&mut self, v: bool) {
        self.has_single_reference = v;
    }

    // uint64 ptr = 6;


    pub fn get_ptr(&self) -> u64 {
        self.ptr
    }
    pub fn clear_ptr(&mut self) {
        self.ptr = 0;
    }

    // Param is passed by value, moved
    pub fn set_ptr(&mut self, v: u64) {
        self.ptr = v;
    }
}

impl ::protobuf::Message for AllocationDescription {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.requested_bytes = tmp;
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.allocated_bytes = tmp;
                },
                3 => {
                    ::protobuf::rt::read_singular_proto3_string_into(wire_type, is, &mut self.allocator_name)?;
                },
                4 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.allocation_id = tmp;
                },
                5 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_bool()?;
                    self.has_single_reference = tmp;
                },
                6 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_uint64()?;
                    self.ptr = tmp;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if self.requested_bytes != 0 {
            my_size += ::protobuf::rt::value_size(1, self.requested_bytes, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.allocated_bytes != 0 {
            my_size += ::protobuf::rt::value_size(2, self.allocated_bytes, ::protobuf::wire_format::WireTypeVarint);
        }
        if !self.allocator_name.is_empty() {
            my_size += ::protobuf::rt::string_size(3, &self.allocator_name);
        }
        if self.allocation_id != 0 {
            my_size += ::protobuf::rt::value_size(4, self.allocation_id, ::protobuf::wire_format::WireTypeVarint);
        }
        if self.has_single_reference != false {
            my_size += 2;
        }
        if self.ptr != 0 {
            my_size += ::protobuf::rt::value_size(6, self.ptr, ::protobuf::wire_format::WireTypeVarint);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        if self.requested_bytes != 0 {
            os.write_int64(1, self.requested_bytes)?;
        }
        if self.allocated_bytes != 0 {
            os.write_int64(2, self.allocated_bytes)?;
        }
        if !self.allocator_name.is_empty() {
            os.write_string(3, &self.allocator_name)?;
        }
        if self.allocation_id != 0 {
            os.write_int64(4, self.allocation_id)?;
        }
        if self.has_single_reference != false {
            os.write_bool(5, self.has_single_reference)?;
        }
        if self.ptr != 0 {
            os.write_uint64(6, self.ptr)?;
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &::std::any::Any {
        self as &::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut ::std::any::Any {
        self as &mut ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        Self::descriptor_static()
    }

    fn new() -> AllocationDescription {
        AllocationDescription::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "requested_bytes",
                    |m: &AllocationDescription| { &m.requested_bytes },
                    |m: &mut AllocationDescription| { &mut m.requested_bytes },
                ));
                fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "allocated_bytes",
                    |m: &AllocationDescription| { &m.allocated_bytes },
                    |m: &mut AllocationDescription| { &mut m.allocated_bytes },
                ));
                fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "allocator_name",
                    |m: &AllocationDescription| { &m.allocator_name },
                    |m: &mut AllocationDescription| { &mut m.allocator_name },
                ));
                fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "allocation_id",
                    |m: &AllocationDescription| { &m.allocation_id },
                    |m: &mut AllocationDescription| { &mut m.allocation_id },
                ));
                fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeBool>(
                    "has_single_reference",
                    |m: &AllocationDescription| { &m.has_single_reference },
                    |m: &mut AllocationDescription| { &mut m.has_single_reference },
                ));
                fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeUint64>(
                    "ptr",
                    |m: &AllocationDescription| { &m.ptr },
                    |m: &mut AllocationDescription| { &mut m.ptr },
                ));
                ::protobuf::reflect::MessageDescriptor::new::<AllocationDescription>(
                    "AllocationDescription",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }

    fn default_instance() -> &'static AllocationDescription {
        static mut instance: ::protobuf::lazy::Lazy<AllocationDescription> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const AllocationDescription,
        };
        unsafe {
            instance.get(AllocationDescription::new)
        }
    }
}

impl ::protobuf::Clear for AllocationDescription {
    fn clear(&mut self) {
        self.requested_bytes = 0;
        self.allocated_bytes = 0;
        self.allocator_name.clear();
        self.allocation_id = 0;
        self.has_single_reference = false;
        self.ptr = 0;
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for AllocationDescription {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for AllocationDescription {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n6tensorflow/core/framework/allocation_description.proto\x12\ntensorflo\
    w\"\xf9\x01\n\x15AllocationDescription\x12'\n\x0frequested_bytes\x18\x01\
    \x20\x01(\x03R\x0erequestedBytes\x12'\n\x0fallocated_bytes\x18\x02\x20\
    \x01(\x03R\x0eallocatedBytes\x12%\n\x0eallocator_name\x18\x03\x20\x01(\t\
    R\rallocatorName\x12#\n\rallocation_id\x18\x04\x20\x01(\x03R\x0callocati\
    onId\x120\n\x14has_single_reference\x18\x05\x20\x01(\x08R\x12hasSingleRe\
    ference\x12\x10\n\x03ptr\x18\x06\x20\x01(\x04R\x03ptrB{\n\x18org.tensorf\
    low.frameworkB\x1bAllocationDescriptionProtosP\x01Z=github.com/tensorflo\
    w/tensorflow/tensorflow/go/core/framework\xf8\x01\x01b\x06proto3\
";

static mut file_descriptor_proto_lazy: ::protobuf::lazy::Lazy<::protobuf::descriptor::FileDescriptorProto> = ::protobuf::lazy::Lazy {
    lock: ::protobuf::lazy::ONCE_INIT,
    ptr: 0 as *const ::protobuf::descriptor::FileDescriptorProto,
};

fn parse_descriptor_proto() -> ::protobuf::descriptor::FileDescriptorProto {
    ::protobuf::parse_from_bytes(file_descriptor_proto_data).unwrap()
}

pub fn file_descriptor_proto() -> &'static ::protobuf::descriptor::FileDescriptorProto {
    unsafe {
        file_descriptor_proto_lazy.get(|| {
            parse_descriptor_proto()
        })
    }
}