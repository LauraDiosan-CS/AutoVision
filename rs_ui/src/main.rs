use iced::futures::channel::mpsc;
use iced::futures::SinkExt;
use iced::widget::{column, container, image, text};
use iced::{futures, Alignment, Element, Length, Task};
use std::thread;
use std::time::Duration;

fn main() -> iced::Result {
    iced::application("AutoVision", Counter::update, Counter::view)
        .run_with(Counter::new)
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
            image::Image::new(self.current_image.clone())
                .width(Length::Fixed(100.0)),
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

fn image_producer() -> impl futures::Stream<Item = Message> {
    let (mut sender, receiver) = mpsc::channel(1);

    thread::spawn(move || {
        let image1 = image::Handle::from_path("assets/ferris.png");
        let image2 = image::Handle::from_path("assets/ferris2.png");
        let mut is_image1 = true;

        loop {
            thread::sleep(Duration::from_secs(1));

            let image_to_send = if is_image1 {
                image1.clone()
            } else {
                image2.clone()
            };
            is_image1 = !is_image1;

            let result = futures::executor::block_on(sender.send(Message::NewData(image_to_send)));

            if result.is_err() {
                break;
            }
        }
    });

    receiver
}
