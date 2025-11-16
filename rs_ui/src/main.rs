use iced::futures::SinkExt;
use iced::futures::channel::mpsc;
use iced::widget::{column, container, image, text};
use iced::{Alignment, Element, Length, Task, futures};
use rs_ipc::container::message::SharedMessage;
use rs_ipc::primitives::memory_mapper::SharedMemoryMapper;
use std::thread;

fn main() -> iced::Result {
    iced::application("AutoVision", Counter::update, Counter::view).run_with(Counter::new)
}

struct Counter {
    frame: u64,
    current_image: image::Handle,
}

#[derive(Debug, Clone)]
pub enum Message {
    NewData(image::Handle),
}

impl Counter {
    fn new() -> (Self, Task<Message>) {
        let initial_state = Self {
            frame: 0,
            current_image: image::Handle::from_path("assets/ferris.png"),
        };

        (initial_state, Task::stream(image_producer()))
    }

    fn update(&mut self, message: Message) {
        match message {
            Message::NewData(image) => {
                self.frame += 1;
                self.current_image = image;
            }
        }
    }

    fn view(&'_ self) -> Element<'_, Message> {
        let content = column![
            text(format!("Frame: {}", self.frame)).size(50),
            image::Image::new(self.current_image.clone()).width(Length::Fixed(100.0)),
        ]
        .padding(20)
        .align_x(Alignment::Center)
        .spacing(10);

        container(content)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into()
    }
}

#[repr(C)]
struct PythonData {
    id: u64,
    width: u32,
    height: u32,
    pixels: [u8],
}

impl PythonData {
    fn size_of_fields() -> usize {
        #[repr(C)]
        struct PythonDataSized {
            id: u64,
            width: u32,
            height: u32,
        }
        size_of::<PythonDataSized>()
    }

    unsafe fn cast_from_u8(data: &[u8]) -> Option<&Self> {
        let payload_size = data.len().checked_sub(Self::size_of_fields())?;
        let slice_ptr: *const [u8] =
            std::ptr::slice_from_raw_parts(data.as_ptr().cast(), payload_size);
        unsafe { (slice_ptr as *const Self).as_ref() }
    }
}

struct RawImageData {
    id: u64,
    width: u32,
    height: u32,
    pixels: Box<[u8]>,
}

fn image_producer() -> impl futures::Stream<Item = Message> {
    let (mut sender, receiver) = mpsc::channel(1);

    thread::spawn(move || {
        let shared_memory =
            unsafe { SharedMemoryMapper::<SharedMessage>::create(c"test".into(), 10 * 1024 * 1024).unwrap() };
        shared_memory.add_reader();
        let mut last_read_version = 0;

        while !shared_memory.is_stopped() {
            let mut result = None;
            
            shared_memory.blocking_read(last_read_version, |new_version, data| {
                last_read_version = new_version;
                let Some(python_data) = (unsafe { PythonData::cast_from_u8(data) }) else {
                    eprintln!("Failed to cast Python data");
                    return;
                };

                result = Some(RawImageData {
                    id: python_data.id,
                    width: python_data.width,
                    height: python_data.height,
                    pixels: Box::from(&python_data.pixels),
                });
            });
            
            let Some(result) = result else {
                continue;
            };
            
            let handle = image::Handle::from_rgba(result.width, result.height, result.pixels);

            if futures::executor::block_on(sender.send(Message::NewData(handle))).is_err() {
                break;
            }
        }
    });

    receiver
}
