import javax.swing.*;
import java.awt.*;
import java.util.Random;

public class RedBall extends Ball{
    Random random = new Random();

    public RedBall(int xSpeed, int ySpeed, int ballSize) {
        super(Color.RED, xSpeed, ySpeed, ballSize);
    }

    @Override
    public void update(char keyChar) {
//        System.out.println("RedBall update");
        switch (keyChar) {
            case 'a':
                this.setXSpeed(-random.nextInt(3) - 1);
                break;
            case 'd':
                this.setXSpeed(random.nextInt(3) + 1);
                break;
            case 'w':
                this.setYSpeed(-random.nextInt(3) - 1);
                break;
            case 's':
                this.setYSpeed(random.nextInt(3) + 1);
        }
    }

    @Override
    public void update(Ball whiteBall) {
        if (this.isIntersect(whiteBall)){
            this.setXSpeed(whiteBall.getXSpeed());
            this.setYSpeed(whiteBall.getYSpeed());
        }
    }
}
